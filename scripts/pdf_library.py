#!/usr/bin/env python3
"""One Git-tracked shelf for finished report PDFs, so a reset cannot eat a deliverable.

Everything the framework writes under ``reports/`` is git-ignored on purpose — a run is
reproducible, so the bytes are not worth a commit. That is exactly wrong for the *finished*
report: an agent-supplied run cannot be reproduced from the repository alone, and the sandbox
has twice wiped ``reports/`` (along with ``~/.tradingagents``) while the branch stayed put.
So the typeset PDF you were asked to read disappeared twice even though the tooling that made
it survived.

This script copies the deliverable into ``reports/library/``, which is deliberately
**tracked** (see the ``reports/*`` + ``!reports/library/`` pair in ``.gitignore``): commit it,
push it, and the report comes back with ``git`` instead of having to be re-authored.

Layout::

    reports/library/
        2026-09-15_OKLO_traditional_chinese.pdf   the deliverable
        2026-09-15_OKLO_traditional_chinese.md    its source markdown (re-render the reader)
        manifest.json                             one entry per shelf item
        index.html                                dark-first browsable shelf (generated)
        README.md                                 the same, as a GitHub-renderable table

Only the Traditional-Chinese (繁中) PDF is saved by default — small (~160 KB, font subsets
embedded) and the one actually read. ``--lang`` files other cuts into the same shelf;
``sync`` picks up whatever is on disk under ``reports/``.

Usage
-----
    python scripts/pdf_library.py add reports/OKLO_2026-09-15/OKLO_2026-09-15_繁體中文報告.pdf \
        --ticker OKLO --name "Oklo Inc. (NYSE)" --markdown complete_report_zh.md --commit
    python scripts/pdf_library.py sync            # file every report dir not yet on the shelf
    python scripts/pdf_library.py list            # what is on the shelf
    python scripts/pdf_library.py build           # rebuild index.html / README.md from manifest

``add`` and ``sync`` rebuild the index as they go. Entries are keyed by slug and deduplicated
by content hash, so re-running after a re-render is a no-op or an update, never a duplicate.

Exit status: 0 on success (including "already shelved, unchanged"), 1 on invalid input —
not a PDF, a bad ticker/date/slug, a missing run directory, a PDF too big for Git — and 2 on an
I/O or git failure (the shelf is only useful once it is committed, so that is an error here).
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from urllib.parse import quote

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LIBRARY = REPO_ROOT / "reports" / "library"

# Shelf filenames carry this language tag; "zh" means Traditional Chinese (繁體中文), which is
# what the bilingual deliverables are built from.
LANG_SLUG = {
    "zh": "traditional_chinese",
    "en": "english",
    "bilingual": "bilingual_en_zh",
    "ja": "japanese",
    "ko": "korean",
}
LANG_LABEL = {
    "zh": "繁體中文",
    "en": "English",
    "bilingual": "中英對照",
    "ja": "日本語",
    "ko": "한국어",
}
# Same five levels the PDF cover badge uses (report_to_pdf.RATING_LABELS).
RATING_LABELS = {
    "Buy": "買進",
    "Overweight": "加碼",
    "Hold": "持有",
    "Underweight": "減碼",
    "Sell": "賣出",
    "REVIEW": "待人工複核",
}

TICKER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,15}$")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
SLUG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
RUN_DIR_RE = re.compile(r"^(?P<ticker>[A-Za-z0-9][A-Za-z0-9._-]*)_(?P<date>\d{4}-\d{2}-\d{2})$")
INVERTED_RE = re.compile(r"^(?P<date>\d{4}-\d{2}-\d{2})_(?P<ticker>[A-Za-z0-9][A-Za-z0-9._-]*)")
# Tried in order: the CJK-named report is the canonical 繁中 cut, the ASCII mirror its twin.
ZH_PDF_GLOBS = ("*_繁體中文報告.pdf", "*_traditional_chinese.pdf", "*繁體中文*.pdf")
RATING_MD_RE = re.compile(r"^\*\*Rating\*\*\s*[:：]\s*([A-Za-z]+)", re.M)
MAX_PDF_BYTES = 5 * 1024 * 1024
WARN_LIBRARY_BYTES = 60 * 1024 * 1024


class LibraryError(Exception):
    """Bad input or an unwritable shelf; mapped to a non-zero exit status in main()."""


def repo_rel(path: Path) -> str:
    """Repo-relative posix path when the file lives in the repo, else an absolute path."""
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def sha256_prefix(path: Path, length: int = 12) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:length]


def pdf_metadata(path: Path) -> dict[str, str | int]:
    """Pages / title / rating from the PDF itself, when PyMuPDF is installed.

    The renderer writes ``title`` and ``keywords = "rating: …"`` into the document info, so a
    dropped-in PDF still gets a labelled shelf entry with no sidecar.
    """
    try:
        import pymupdf  # noqa: PLC0415  (optional [pdf] extra)
    except ImportError:
        return {}
    try:
        with pymupdf.open(path) as doc:
            info = doc.metadata or {}
            out: dict[str, str | int] = {"pages": doc.page_count}
            title = (info.get("title") or "").strip()
            if title:
                out["title"] = title
            rating = re.search(r"rating:\s*([A-Za-z]+)", info.get("keywords") or "", re.I)
            if rating:
                out["rating"] = rating.group(1)
        return out
    except Exception:  # a damaged PDF should not sink the whole shelf
        return {}


# ---------------------------------------------------------------------------------------------
# entries
# ---------------------------------------------------------------------------------------------


@dataclass
class Entry:
    """One shelved report: the PDF in ``reports/library/`` plus what a reader needs to know."""

    slug: str
    file: str
    ticker: str
    date: str
    lang: str = "zh"
    name: str = ""
    title: str = ""
    rating: str = ""
    pages: int = 0
    bytes: int = 0
    sha256: str = ""
    added: str = ""
    source: str = ""
    markdown: str = ""
    reader: str = ""
    note: str = ""

    @property
    def label(self) -> str:
        return LANG_LABEL.get(self.lang, self.lang)

    @property
    def rating_label(self) -> str:
        return RATING_LABELS.get(self.rating, self.rating)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict) -> Entry:
        known = {f.name for f in cls.__dataclass_fields__.values()}  # forward-compatible reads
        return cls(**{k: v for k, v in raw.items() if k in known and v is not None})


def derive_slug(*, ticker: str, date_str: str, lang: str) -> str:
    return f"{date_str}_{ticker}_{LANG_SLUG[lang]}"


def company_from_pack(pack: Path) -> str:
    """Best-effort asset name for the shelf card: ``company``, else the pack's context prefix."""
    if not pack.is_file():
        return ""
    try:
        raw = json.loads(pack.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return ""
    name = str(raw.get("company") or "").strip()
    if not name:
        context = str(raw.get("instrument_context") or "")
        name = re.split(r"\s+[\u2014\u2013]\s+", context, maxsplit=1)[0].strip()
    return name[:120]


def derive_rating(markdown: Path | None) -> str:
    """Pull ``**Rating**: Underweight`` out of the portfolio-manager section, if present."""
    if markdown is None or not Path(markdown).is_file():
        return ""
    try:
        match = RATING_MD_RE.search(Path(markdown).read_text(encoding="utf-8"))
    except OSError:
        return ""
    return match.group(1) if match else ""


# ---------------------------------------------------------------------------------------------
# the shelf
# ---------------------------------------------------------------------------------------------


class Library:
    def __init__(self, root: Path | str = DEFAULT_LIBRARY, *, dry_run: bool = False):
        self.root = Path(root)
        self.dry_run = dry_run
        self.manifest_path = self.root / "manifest.json"

    # -- manifest ----------------------------------------------------------------------------

    def load(self) -> list[Entry]:
        if not self.manifest_path.is_file():
            return []
        try:
            data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise LibraryError(f"cannot read {repo_rel(self.manifest_path)}: {exc}") from exc
        entries = [Entry.from_dict(raw) for raw in data.get("entries", [])]
        return self.sort(entries)

    def save(self, entries: list[Entry]) -> None:
        payload = {
            "schema": 1,
            "updated": date.today().isoformat(),
            "note": "Generated by scripts/pdf_library.py — tracked in Git on purpose, so a "
                    "sandbox reset that clears reports/ cannot clear the finished reports.",
            "entries": [entry.to_dict() for entry in entries],
        }
        if self.dry_run:
            return
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    @staticmethod
    def sort(entries: list[Entry]) -> list[Entry]:
        return sorted(entries, key=lambda e: (e.date, e.ticker.upper(), e.lang), reverse=True)

    # -- writing ------------------------------------------------------------------------------

    def add(
        self,
        pdf: Path,
        *,
        ticker: str,
        date_str: str,
        slug: str | None = None,
        lang: str = "zh",
        name: str = "",
        title: str = "",
        rating: str = "",
        markdown: Path | None = None,
        reader: Path | None = None,
        note: str = "",
        force: bool = False,
        persist: bool = True,
    ) -> tuple[Entry, str]:
        """Shelve one PDF (plus optional source markdown); returns (entry, status).

        Status is ``added``, ``updated`` (same slug, different bytes) or ``unchanged``. The
        manifest is rewritten as part of the call (``persist=False`` defers that to the caller),
        so a file that was only copied but never listed would look like a hand-dropped PDF.
        """
        pdf = Path(pdf)
        if not pdf.is_file():
            raise LibraryError(f"no such PDF: {pdf}")
        with pdf.open("rb") as handle:
            if handle.read(5) != b"%PDF-":
                raise LibraryError(f"{pdf} is not a PDF (missing %PDF- header)")
        ticker = (ticker or "").strip().upper()
        if not TICKER_RE.match(ticker):
            raise LibraryError(f"bad ticker {ticker!r}: use e.g. OKLO, RKLB, BRK.B")
        if not DATE_RE.match(date_str or ""):
            raise LibraryError(f"bad date {date_str!r}: use YYYY-MM-DD")
        if lang not in LANG_SLUG:
            raise LibraryError(f"unknown language {lang!r}: use one of {', '.join(sorted(LANG_SLUG))}")
        slug = slug or derive_slug(ticker=ticker, date_str=date_str, lang=lang)
        if not SLUG_RE.match(slug):
            raise LibraryError(f"bad slug {slug!r}: keep it to letters, digits, dot, dash, underscore")
        size = pdf.stat().st_size
        if size > MAX_PDF_BYTES:
            raise LibraryError(
                f"{pdf.name} is {size / 1e6:.1f} MB — too big for a tracked shelf; "
                "link it from reports/index.html instead (HTML readers are regenerable)"
            )

        digest = sha256_prefix(pdf)
        entries = self.load()
        prior = next((e for e in entries if e.slug == slug), None)
        if prior and prior.sha256 == digest and not force:
            return prior, "unchanged"

        meta = pdf_metadata(pdf)
        entry = Entry(
            slug=slug,
            file=f"{slug}.pdf",
            ticker=ticker,
            date=date_str,
            lang=lang,
            name=name,
            title=title or str(meta.get("title") or ""),
            rating=rating or str(meta.get("rating") or "") or derive_rating(markdown),
            pages=int(meta.get("pages") or (prior.pages if prior else 0)),
            bytes=size,
            sha256=digest,
            added=(prior.added if prior and prior.added else date.today().isoformat()),
            source=repo_rel(pdf),
            markdown="" if markdown is None else repo_rel(Path(markdown)),
            reader="" if reader is None else repo_rel(Path(reader)),
            note=note,
        )

        if not self.dry_run:
            self.root.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(pdf, self.root / entry.file)
            if markdown is not None and Path(markdown).is_file():
                shutil.copyfile(markdown, self.root / f"{slug}.md")
        status = "updated" if prior else "added"
        if persist and status != "unchanged":
            self.put_entry(entry)
        return entry, status

    def put_entry(self, entry: Entry) -> None:
        """Insert/replace one entry in the manifest (used by add/sync, keeps sort order)."""
        entries = [e for e in self.load() if e.slug != entry.slug] + [entry]
        self.save(self.sort(entries))

    def prune(self) -> list[Entry]:
        """Drop manifest entries whose PDF is no longer on the shelf."""
        entries = self.load()
        kept = [e for e in entries if (self.root / e.file).is_file()]
        if len(kept) != len(entries):
            self.save(kept)
            return [e for e in entries if e not in kept]
        return []

    def adopt_untracked(self) -> list[Entry]:
        """Describe PDFs sitting on the shelf that the manifest does not know about."""
        entries = {e.file: e for e in self.load()}
        added: list[Entry] = []
        for path in sorted(self.root.glob("*.pdf")):
            if path.name in entries:
                continue
            hit = RUN_DIR_RE.match(path.stem) or INVERTED_RE.match(path.stem)
            ticker = (hit.group("ticker") if hit else "") or path.stem.split("_")[0]
            date_str = hit.group("date") if hit else ""
            meta = pdf_metadata(path)
            added.append(Entry(
                slug=path.stem,
                file=path.name,
                ticker=ticker,
                date=date_str,
                lang="bilingual" if "bilingual" in path.stem else ("en" if "english" in path.stem else "zh"),
                title=str(meta.get("title") or ""),
                rating=str(meta.get("rating") or ""),
                pages=int(meta.get("pages") or 0),
                bytes=path.stat().st_size,
                sha256=sha256_prefix(path),
                added=date.today().isoformat(),
                note="手動放入（manually dropped in; metadata inferred）",
            ))
        if added:
            self.save(self.sort(self.load() + added))
        return added

    # -- reading ------------------------------------------------------------------------------

    def total_bytes(self, entries: list[Entry]) -> int:
        return sum(e.bytes for e in entries)

    def sync(self, reports_dir: Path) -> list[tuple[Entry, str]]:
        """Shelve the 繁中 PDF of every ``reports/<TICKER>_<DATE>/`` run not already on the shelf."""
        reports_dir = Path(reports_dir)
        if not reports_dir.is_dir():
            raise LibraryError(f"no such reports directory: {reports_dir}")
        done: list[tuple[Entry, str]] = []
        for run in sorted(p for p in reports_dir.iterdir() if p.is_dir()):
            hit = RUN_DIR_RE.match(run.name)
            if not hit:
                continue
            pdf = next((p for pat in ZH_PDF_GLOBS for p in sorted(run.glob(pat)) if p.is_file()), None)
            if pdf is None:
                continue
            markdown = next((run / n for n in ("complete_report_zh.md", "complete_report.md")
                             if (run / n).is_file()), None)
            name = company_from_pack(run / "pack.json")
            reader = pdf.with_suffix(".html")
            try:
                entry, status = self.add(
                    pdf,
                    ticker=hit.group("ticker"),
                    date_str=hit.group("date"),
                    name=name,
                    markdown=markdown,
                    reader=reader if reader.is_file() else None,
                    note="由 sync 自動放入（auto-filed by sync）",
                )
            except LibraryError as exc:
                print(f"skip {repo_rel(pdf)}: {exc}", file=sys.stderr)
                continue
            done.append((entry, status))
        return done

    # -- rendering ----------------------------------------------------------------------------

    def render_index(self, entries: list[Entry]) -> str:
        total = self.total_bytes(entries)
        cards = "\n".join(self._card(e) for e in entries) or self._empty()
        return f"""<!doctype html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TradingAgents · 報告 PDF 書架 (report PDF library)</title>
<style>
{_INDEX_CSS}
</style>
</head>
<body>
<header class="bar">
  <strong style="font-size:13px">TradingAgents · reports/library</strong>
  <span class="grow"></span>
  <span class="seg">
    <label><input type="radio" name="theme" id="theme-auto" checked><span>自動 Auto</span></label>
    <label><input type="radio" name="theme" id="theme-dark"><span>深色 Dark</span></label>
    <label><input type="radio" name="theme" id="theme-light"><span>淺色 Light</span></label>
  </span>
</header>

<div class="wrap">
  <h1>報告 PDF 書架 · the shelf that survives a sandbox reset</h1>
  <p class="sub">{len(entries)} 份報告 · 共 {total / 1024:.0f} KB ·
  更新 {html.escape(date.today().isoformat())} ·
  此目錄<strong>刻意納入 Git 追蹤</strong>，重置後 <code>git checkout</code>／<code>git pull</code> 即全部回來了</p>

  <ul>
{cards}
  </ul>

  <div class="note">
    <h3>若 Chrome 仍然擋下 PDF（沙箱 iframe 無法內嵌 PDF 外掛）</h3>
    <ol>
      <li>點 <strong>下載 Download</strong>：存到自己的資料夾後用系統檢視器開啟；PDF 本身是完整排版、含內嵌繁中字型。</li>
      <li>點 <strong>Markdown</strong>：純文字全文，瀏覽器直接可讀，不需要任何外掛。</li>
      <li>想要「分頁＋圖片式」閱讀器：對上面的 Markdown 執行
      <code>.venv/bin/python scripts/report_to_pdf.py &lt;markdown&gt; --html</code> 一句重建
      （閱讀器檔案較大、刻意不入 Git，故重置後才需要重建）。</li>
    </ol>
  </div>
</div>

<footer>
  產製：<code>scripts/agent_supplied_run.py</code>（免 API 金鑰的離線框架執行）→
  <code>scripts/report_to_pdf.py</code>（PyMuPDF 排版：封面決策徽章、目錄書籤、表格跨頁重複表頭）；
  書架：<code>scripts/pdf_library.py add</code>／<code>sync</code>。
  <strong>僅供研究使用，不構成投資建議。</strong>
</footer>
</body>
</html>
"""

    def rel_href(self, stored: str) -> str:
        """href from the shelf directory to a repo path recorded by ``repo_rel``."""
        target = Path(stored)
        if not target.is_absolute():
            target = REPO_ROOT / stored
        try:
            return os.path.relpath(target, self.root).replace(os.sep, "/")
        except ValueError:  # pragma: no cover - different Windows drive
            return str(target).replace(os.sep, "/")

    def _card(self, entry: Entry) -> str:
        rating = (f'<span class="chip rate">最終決策：{html.escape(entry.rating_label)}'
                  f' {html.escape(entry.rating)}</span>' if entry.rating else "")
        title = entry.title or f"{entry.ticker} 交易分析報告"
        bits = [f'<span class="chip">{entry.label}</span>',
                f'<span class="chip">{html.escape(entry.date)}</span>']
        if entry.pages:
            bits.append(f'<span class="chip">{entry.pages} 頁</span>')
        bits.append(f'<span class="chip">{entry.bytes / 1024:.0f} KB</span>')
        bits.append(f'<span class="chip mono">sha256:{html.escape(entry.sha256[:8])}</span>')
        actions = [f'<a class="btn primary" href="{quote(entry.file)}" download'
                   f'>下載 PDF Download</a>',
                   f'<a class="btn" href="{quote(entry.file)}" target="_blank" '
                   f'rel="noopener">新分頁開啟</a>']
        if entry.markdown and (self.root / f"{entry.slug}.md").is_file():
            actions.append(f'<a class="btn" href="{quote(f"{entry.slug}.md")}" '
                           f'download>Markdown · 全文</a>')
        if entry.reader:
            actions.append(f'<a class="btn ghost" href="{quote(self.rel_href(entry.reader))}"'
                           f' target="_blank" rel="noopener">閱讀器（重置後需重建）</a>')
        meta = []
        if entry.name:
            meta.append(html.escape(entry.name))
        if entry.source:
            meta.append(f"來源 <code>{html.escape(entry.source)}</code>")
        if entry.note:
            meta.append(html.escape(entry.note))
        return f"""    <li class="card">
      <div class="name">{html.escape(title)} {rating}</div>
      <div class="chips">{''.join(bits)}</div>
      <p>{' · '.join(meta)}</p>
      <div class="actions">{''.join(actions)}</div>
    </li>"""

    def _empty(self) -> str:
        return ('    <li class="card"><div class="name">書架上還沒有報告</div>'
                '<p>跑完報告後執行 <code>python scripts/pdf_library.py sync</code>，'
                '或 <code>add &lt;pdf&gt; --ticker XXXX --date YYYY-MM-DD</code>。</p></li>')

    def render_readme(self, entries: list[Entry]) -> str:
        rows = ["| 日期 | 股票 | 語言 | 決策 | 頁 | 檔案 |",
                "|---|---|---|---|---|---|"]
        for e in entries:
            link = f"[`{html.escape(e.file)}`]({quote(e.file)})"
            md = f" · [markdown]({quote(f'{e.slug}.md')})" if (self.root / f"{e.slug}.md").is_file() else ""
            rows.append(f"| {e.date} | {html.escape(e.ticker)} | {e.label} | "
                        f"{html.escape(f'{e.rating} {e.rating_label}'.strip())} | {e.pages or '—'} | "
                        f"{link}{md} |")
        body = "\n".join(rows) if entries else "_尚未放入任何報告 — 執行 `python scripts/pdf_library.py sync`。_"
        return f"""# 報告 PDF 書架 · report PDF library

Every finished TradingAgents report PDF that is worth keeping lives here. This folder is
**tracked in Git on purpose** (the rest of `reports/` is ignored), so a sandbox reset that
clears `reports/` cannot clear the deliverables — they come back with the branch.

Default cut per ticker: the 純繁體中文 PDF. Re-file or add with:

```bash
python scripts/pdf_library.py sync            # every reports/<TICKER>_<DATE>/ not yet shelved
python scripts/pdf_library.py add <pdf> --ticker OKLO --date 2026-09-15 --markdown <zh.md>
python scripts/pdf_library.py build           # regenerate this table and index.html
```

{body}

`manifest.json` holds the machine-readable version of this table.
研究用，不構成投資建議。
"""

    def build(self) -> dict[str, int | str]:
        """Regenerate ``index.html`` + ``README.md`` from the shelf contents."""
        adopted = self.adopt_untracked()
        removed = self.prune()
        entries = self.load()
        out = {"entries": len(entries), "adopted": len(adopted), "pruned": len(removed)}
        if self.dry_run:
            return out
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "index.html").write_text(self.render_index(entries), encoding="utf-8")
        (self.root / "README.md").write_text(self.render_readme(entries), encoding="utf-8")
        out["index"] = repo_rel(self.root / "index.html")
        total = self.total_bytes(entries)
        if total > WARN_LIBRARY_BYTES:
            out["warning"] = (f"shelf is {total / 1e6:.1f} MB — consider pruning old entries "
                              "(git history keeps them anyway, so a squash-merge helps)")
        return out


_INDEX_CSS = """
:root {
  color-scheme: light dark;
  --bg: #f2f4f7; --fg: #161a20; --muted: #5c6572; --card: #fff; --line: #d5dbe3;
  --accent: #0d5c73; --shadow: 0 1px 3px rgba(16,24,32,.12);
}
@media (prefers-color-scheme: dark) {
  :root { --bg: #0d1014; --fg: #e8ecf2; --muted: #9aa7b6; --card: #161b21; --line: #2a333d;
          --accent: #56b6d6; --shadow: 0 1px 4px rgba(0,0,0,.5); }
}
body:has(#theme-dark:checked)  { color-scheme: dark;  --bg:#0d1014; --fg:#e8ecf2; --muted:#9aa7b6;
  --card:#161b21; --line:#2a333d; --accent:#56b6d6; --shadow:0 1px 4px rgba(0,0,0,.5); }
body:has(#theme-light:checked) { color-scheme: light; --bg:#f2f4f7; --fg:#161a20; --muted:#5c6572;
  --card:#fff; --line:#d5dbe3; --accent:#0d5c73; --shadow:0 1px 3px rgba(16,24,32,.12); }
* { box-sizing: border-box; }
body { margin: 0; padding: 0 0 64px; background: var(--bg); color: var(--fg);
  font: 14.5px/1.6 ui-sans-serif, -apple-system, "Segoe UI", "Noto Sans TC", sans-serif; }
.bar { position: sticky; top: 0; z-index: 5; display: flex; gap: 12px; align-items: center; flex-wrap: wrap;
  padding: 10px 18px; background: var(--card); border-bottom: 1px solid var(--line); box-shadow: var(--shadow); }
.seg { display: inline-flex; border: 1px solid var(--line); border-radius: 9px; overflow: hidden; }
.seg label { padding: 4px 9px; font-size: 12px; color: var(--muted); cursor: pointer; }
.seg label + label { border-left: 1px solid var(--line); }
.seg input { display: none; }
.seg label:has(input:checked) { background: color-mix(in srgb, var(--accent) 18%, transparent); color: var(--fg); font-weight: 650; }
.grow { flex: 1 1 auto; }
.wrap { max-width: 900px; margin: 0 auto; padding: 28px 18px 0; }
h1 { font-size: 20px; line-height: 1.3; margin: 0 0 4px; }
.sub { color: var(--muted); margin: 0 0 20px; font-size: 13.5px; }
.chips { display: flex; gap: 7px; flex-wrap: wrap; margin: 7px 0 0; }
.chip { font-size: 11.8px; padding: 2px 9px; border-radius: 999px; border: 1px solid var(--line); color: var(--muted); }
.chip.mono { font-family: ui-monospace, Menlo, monospace; }
.chip.rate { color: var(--fg); font-weight: 650; border-color: color-mix(in srgb, var(--accent) 55%, transparent);
  background: color-mix(in srgb, var(--accent) 14%, transparent); }
ul { list-style: none; padding: 0; margin: 0 0 24px; display: grid; gap: 12px; }
li.card { border: 1px solid var(--line); border-radius: 13px; background: var(--card); padding: 14px 16px;
  box-shadow: var(--shadow); }
li.card .name { font-weight: 660; font-size: 14.5px; display: flex; gap: 10px; align-items: baseline; flex-wrap: wrap; }
li.card p { margin: 9px 0 11px; color: var(--muted); font-size: 12.6px; }
.actions { display: flex; gap: 8px; flex-wrap: wrap; }
a.btn { font-size: 12.3px; text-decoration: none; color: var(--fg); border: 1px solid var(--line);
  border-radius: 9px; padding: 4px 10px; display: inline-flex; gap: 6px; align-items: center; }
a.btn:hover { border-color: var(--accent); color: var(--accent); }
a.btn.primary { background: var(--accent); border-color: var(--accent); color: #fff; font-weight: 650; }
a.btn.primary:hover { color: #fff; filter: brightness(1.08); }
a.btn.ghost { color: var(--muted); font-style: italic; }
code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; }
.note { border-left: 3px solid var(--accent); padding: 4px 0 4px 14px; color: var(--muted); font-size: 13px; }
.note h3 { font-size: 13.5px; color: var(--fg); margin: 0 0 6px; }
.note ol { margin: 0; padding-left: 20px; }
.note li { margin-bottom: 5px; }
footer { max-width: 900px; margin: 26px auto 0; padding: 0 18px; color: var(--muted); font-size: 12px; }
@media print { .bar, .note { display: none !important; } body { background: #fff; } }
"""


# ---------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--library", type=Path, default=DEFAULT_LIBRARY,
                        help=f"shelf directory (default: {repo_rel(DEFAULT_LIBRARY)})")
    parser.add_argument("--dry-run", action="store_true", help="describe the change, write nothing")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="pdf_library.py",
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_add = sub.add_parser("add", help="shelve one PDF (+ its markdown source)")
    _common(p_add)
    p_add.add_argument("pdf", type=Path)
    p_add.add_argument("--ticker", required=True)
    p_add.add_argument("--date", dest="date_str", help="YYYY-MM-DD (default: today)")
    p_add.add_argument("--lang", default="zh", choices=sorted(LANG_SLUG))
    p_add.add_argument("--slug", help="shelf stem (default: <date>_<ticker>_<lang>)")
    p_add.add_argument("--name", default="", help="company/asset shown on the card")
    p_add.add_argument("--title", default="", help="report title (default: PDF metadata)")
    p_add.add_argument("--rating", default="", help="PM decision (default: PDF metadata / markdown)")
    p_add.add_argument("--markdown", type=Path, help="source markdown to keep beside the PDF")
    p_add.add_argument("--reader", type=Path, help="HTML reader to link (kept outside the shelf)")
    p_add.add_argument("--note", default="")
    p_add.add_argument("--force", action="store_true", help="re-shelve even if the hash matches")
    p_add.add_argument("--no-index", action="store_true", help="skip rebuilding index.html")
    p_add.add_argument("--commit", action="store_true",
                       help="git add + commit the shelf (nothing is durable until it is pushed)")

    p_sync = sub.add_parser("sync", help="shelve every reports/<TICKER>_<DATE>/ not yet on the shelf")
    _common(p_sync)
    p_sync.add_argument("--reports", type=Path, default=REPO_ROOT / "reports")
    p_sync.add_argument("--commit", action="store_true", help=argparse.SUPPRESS)

    p_list = sub.add_parser("list", help="print the shelf")
    _common(p_list)
    p_list.add_argument("--json", action="store_true")

    p_build = sub.add_parser("build", help="regenerate index.html + README.md from the shelf")
    _common(p_build)
    p_build.add_argument("--commit", action="store_true", help=argparse.SUPPRESS)
    return ap


def _git_commit(library: Library, message: str) -> str:
    """Stage the shelf and commit it (never pushes — pushing is the user's call)."""
    try:
        subprocess.run(["git", "rev-parse", "--is-inside-work-tree"], cwd=REPO_ROOT, check=True,
                       capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "not a git work tree — skipped"
    subprocess.run(["git", "add", "-A", "--", str(library.root)], cwd=REPO_ROOT, check=True,
                   capture_output=True)
    staged = subprocess.run(["git", "diff", "--cached", "--name-only"], cwd=REPO_ROOT,
                            capture_output=True, text=True).stdout.split()
    if not staged:
        return "nothing to commit (shelf already up to date)"
    subprocess.run(["git", "commit", "-q", "-m", message, "--", *staged], cwd=REPO_ROOT,
                   check=True, capture_output=True)
    return f"committed {len(staged)} file(s) — 記得上 push 才不會被重置洗掉"


def _report(entry: Entry, status: str, root: Path | None = None) -> None:
    pages = f"{entry.pages}p · " if entry.pages else ""
    shown = Path(root or DEFAULT_LIBRARY) / entry.file
    print(f"{status:9s} {entry.ticker} {entry.date} {entry.label} · {pages}"
          f"{entry.bytes / 1024:.0f} KB · {repo_rel(shown)}"
          + (f" · {entry.rating}/{entry.rating_label}" if entry.rating else ""))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    library = Library(args.library, dry_run=args.dry_run)
    try:
        if args.cmd == "add":
            date_str = args.date_str or date.today().isoformat()
            entry, status = library.add(
                args.pdf, ticker=args.ticker, date_str=date_str, slug=args.slug, lang=args.lang,
                name=args.name, title=args.title, rating=args.rating, markdown=args.markdown,
                reader=args.reader, note=args.note, force=args.force,
            )
            _report(entry, status, library.root)
            if not args.no_index:
                stats = library.build()
                if not args.dry_run:
                    print(f"index  {stats['entries']} 份 · {stats['index']}")
            if args.commit and not args.dry_run and status != "unchanged":
                print(f"git    {_git_commit(library, f'Shelve {entry.slug} report PDF')}")
        elif args.cmd == "sync":
            done = library.sync(args.reports)
            for entry, status in done:
                _report(entry, status, library.root)
            if not done:
                print("sync   沒有新報告（nothing new under reports/）")
            stats = library.build()
            if not args.dry_run:
                print(f"index  {stats['entries']} 份 · {stats['index']}")
            if args.commit and not args.dry_run and any(s != "unchanged" for _, s in done):
                print(f"git    {_git_commit(library, 'Shelve regenerated report PDFs')}")
        elif args.cmd == "list":
            entries = library.load()
            if args.json:
                print(json.dumps([e.to_dict() for e in entries], ensure_ascii=False, indent=2))
            elif not entries:
                print("(empty shelf)")
            else:
                total = library.total_bytes(entries)
                print(f"{len(entries)} 份報告 · {total / 1024:.0f} KB · {repo_rel(library.root)}")
                for entry in entries:
                    _report(entry, "·", library.root)
        elif args.cmd == "build":
            stats = library.build()
            for key in ("adopted", "pruned"):
                if stats[key]:
                    print(f"{key:8s} {stats[key]}")
            print(f"{stats['entries']} 份報告 · {stats['index']}")
            if "warning" in stats:
                print(f"warning {stats['warning']}", file=sys.stderr)
            if args.commit and not args.dry_run:
                print(f"git    {_git_commit(library, 'Rebuild the report PDF library index')}")
    except LibraryError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
