#!/usr/bin/env python3
"""Render a TradingAgents report tree (or any markdown report) to a typeset PDF.

Built for non-Latin output: a run localised with ``TRADINGAGENTS_OUTPUT_LANGUAGE`` (e.g.
繁體中文, 日本語, 한국어) cannot be typeset with the base-14 PDF fonts, and the usual
markdown→PDF routes (pandoc/LaTeX, weasyprint) drag in a TeX tree or Pango/Cairo system
libraries that a bare machine or container does not have. This exporter needs only PyMuPDF:
it uses MuPDF's built-in pan-CJK face (``china-t``/``china-s``/``japan``/``korea`` → Droid
Sans Fallback), which carries Simplified *and* Traditional Han glyphs as separate codepoints
— so 實 and 实 never substitute for one another — and embeds them in the file. Pass
``--font`` (with ``--font-bold`` for a true bold face) to typeset with a real family such as
Noto Sans TC instead.

Input is the report the framework already writes: ``complete_report.md`` produced by
:func:`tradingagents.reporting.write_report_tree`. The renderer reproduces its markdown
structure as a paginated document — cover page carrying the 5-tier rating extracted by the
repo's own :func:`extract_rating`, section headings, tables whose header row repeats across
page breaks, block quotes, fenced code, rules, a PDF outline, and a running footer with page
numbers.

Markdown coverage is the subset the report tree emits: ATX headings, paragraphs, ``-``/``*``/
``1.`` lists, ``|`` tables with a ``---`` separator row, ``>`` quotes, ``` fences, ``---``
rules, and inline ``**bold**`` / ``*italic*`` / `` `code` `` / ``[text](url)``. Anything else
degrades to plain text instead of failing, since the input is machine-generated.

Base-14 faces (Helvetica/Courier) are avoided for body text on purpose: they cannot encode the
typographic symbols analysts' prose is full of (≈ → ± ½ — ), and MuPDF raises rather than
substituting. Droid Sans Fallback covers both those symbols and Latin, so one face serves every
language; ``cour`` is used only for pure-ASCII code spans.

Usage
-----
    python scripts/report_to_pdf.py REPORT.md --out REPORT.pdf
    python scripts/report_to_pdf.py complete_report.md --out RKLB_繁體中文.pdf --lang zh \
        --font /usr/share/fonts/opentype/noto/NotoSansCJK.ttc --dpi-preview 120

``--lang`` (default ``auto``, detected from the report's script) switches only the strings the
renderer itself prints — cover caption, footer page count, rating label. Report prose is
whatever language the run produced: translate it upstream (``TRADINGAGENTS_OUTPUT_LANGUAGE``)
or in the markdown, never here. :func:`render_pdf` is the programmatic entry point.
"""

from __future__ import annotations

import argparse
import contextlib
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import pymupdf as fitz
except ImportError:
    print(
        'error: PyMuPDF is required for PDF output. Install it with\n'
        '    pip install "tradingagents[pdf]"        (or: pip install pymupdf)',
        file=sys.stderr,
    )
    raise SystemExit(2) from None

from tradingagents.agents.utils.rating import extract_rating  # noqa: E402

PAGE_SIZES = {"a4": (595.28, 841.89), "letter": (612.0, 792.0), "a5": (419.53, 595.28)}

# One embeddable, symbol-complete face per language; MuPDF ships these in every wheel.
BUILTIN_FACES = {"en": "china-t", "zh": "china-t", "ja": "japan", "ko": "korea"}
UI: dict[str, dict[str, str]] = {
    "en": {
        "cover": "Multi-agent trading analysis",
        "rating": "Final decision",
        "page": "Page %s",
        "of": "of %s",
        "sections": "Contents",
        "colon": ": ",
    },
    "zh": {
        "cover": "多代理交易分析報告",
        "rating": "最終決策",
        "page": "第 %s 頁",
        "of": "／共 %s 頁",
        "sections": "目錄",
        "colon": "：",
    },
    "ja": {
        "cover": "マルチエージェント投資分析",
        "rating": "最終判断",
        "page": "%s ページ",
        "of": "／全 %s ページ",
        "sections": "目次",
        "colon": "：",
    },
    "ko": {
        "cover": "멀티 에이전트 매매 분석",
        "rating": "최종 결정",
        "page": "%s 페이지",
        "of": " / 총 %s 페이지",
        "sections": "목차",
        "colon": ": ",
    },
}

RATING_LABELS = {
    "Buy": "買進",
    "Overweight": "加碼",
    "Hold": "持有",
    "Underweight": "減碼",
    "Sell": "賣出",
    "REVIEW": "待人工複核",
}

INLINE_RE = re.compile(r"(\*\*.+?\*\*|`[^`]+`|\*[^*\n]+?\*|\[[^\]]+\]\([^)]*\))", re.S)
TABLE_SEP_RE = re.compile(r"^\s*\|?[\s:|-]*-[\s:|-]*\|?\s*$")
CJK_SOFT = re.compile(r"[\u2e80-\u9fff\uf900-\ufaff\u3000-\u303f\uff00-\uffef]")


@dataclass
class Span:
    text: str
    bold: bool = False
    mono: bool = False


@dataclass
class Line:
    spans: list[Span]
    size: float
    x: float = 0.0
    color: tuple[float, float, float] = (0.09, 0.11, 0.16)
    leading_mult: float = 1.55
    mono_all: bool = False
    rule: bool = False
    rule_color: tuple[float, float, float] = (0.87, 0.89, 0.93)

    @property
    def height(self) -> float:
        return self.size * self.leading_mult


@dataclass
class Block:
    kind: str
    lines: list[Line] = field(default_factory=list)
    header_rows: list[list[list[Span]]] = field(default_factory=list)
    rows: list[list[list[Span]]] = field(default_factory=list)
    columns: list[float] = field(default_factory=list)
    cell_size: float = 0.0
    anchor: str | None = None
    keep_with_next: bool = False


def _parse_inline(text: str) -> list[Span]:
    """Split markdown emphasis/code/links into styled spans."""
    spans: list[Span] = []
    for part in INLINE_RE.split(text):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**") and len(part) > 4:
            spans.append(Span(part[2:-2].strip(), bold=True))
        elif part.startswith("`") and part.endswith("`") and len(part) > 2:
            spans.append(Span(part[1:-1], mono=True))
        elif part.startswith("*") and part.endswith("*") and len(part) > 2:
            spans.append(Span(part[1:-1]))
        elif part.startswith("["):
            match = re.match(r"\[([^\]]+)\]\(([^)]*)\)", part)
            spans.append(Span(match.group(1) if match else part))
        else:
            spans.append(Span(part))
    return [s for s in spans if s.text] or [Span("")]


class Typesetter:
    """Font metrics + wrapping. Wraps Latin per word and Han per character."""

    def __init__(self, regular, bold, mono, faux_bold: bool):
        self.regular, self.bold, self.mono = regular, bold, mono
        self.faux_bold = faux_bold

    def face(self, span: Span) -> Any:
        if span.mono and self.mono is not None and span.text.isascii():
            return self.mono
        if span.bold and not self.faux_bold:
            return self.bold
        return self.regular

    def width(self, span: Span, size: float) -> float:
        return self.face(span).text_length(span.text, fontsize=size)

    def spans_width(self, spans: list[Span], size: float) -> float:
        return sum(self.width(s, size) for s in spans)

    def wrap(self, spans: list[Span], width: float, size: float) -> list[list[Span]]:
        out: list[list[Span]] = []
        cur: list[Span] = []
        used = 0.0

        def flush():
            nonlocal cur, used
            if cur:
                out.append(cur)
                cur, used = [], 0.0

        for span in spans:
            units: list[str] = []
            if CJK_SOFT.search(span.text):
                # Mixed Han/Latin: tokenize into runs so Han breaks per char.
                for token in re.findall(r"\s+|[^\sA-Za-z0-9$%\u2e80-\u9fff]+|[\u2e80-\u9fff]|[\w$%.:\-]+",
                                         span.text):
                    units.append(token)
            else:
                units = re.split(r"(\s+)", span.text)
            for token in units:
                if not token:
                    continue
                pieces = [token] if (token.isspace() or len(token) == 1 or not CJK_SOFT.match(token)) else list(token)
                for piece in pieces:
                    w = self.width(Span(piece, bold=span.bold, mono=span.mono), size)
                    if used + w > width and cur:
                        flush()
                    if piece.isspace() and not cur:
                        continue
                    cur.append(Span(piece, bold=span.bold, mono=span.mono))
                    used += w
        flush()
        return out or [[Span("")]]


def _table_cells(rows: list[list[str]], typeset: Typesetter, width: float, size: float):
    """Column widths from natural content, then wrapped cell lines per row."""
    ncol = max(len(r) for r in rows)
    rows = [r + [""] * (ncol - len(r)) for r in rows]
    natural = []
    for c in range(ncol):
        natural.append(
            max((typeset.spans_width(_parse_inline(rows[r][c]), size) for r in range(len(rows))), default=24.0)
        )
    gap = 12.0
    total = sum(natural) + gap * (ncol - 1)
    if total > width:
        avail = max(width - gap * (ncol - 1), ncol * 34.0)
        scale = avail / max(sum(natural), 1.0)
        natural = [max(34.0, n * scale) for n in natural]
    grid = []
    for row in rows:
        wrapped_row = []
        for c, cell in enumerate(row):
            wrapped_row.append(typeset.wrap(_parse_inline(cell), max(natural[c] - 6, 20.0), size))
        grid.append(wrapped_row)
    return natural, grid


def parse_markdown(text: str, typeset: Typesetter, width: float, size: float) -> list[Block]:
    """Report markdown → blocks of pre-wrapped lines (tables keep their cell grid)."""
    blocks: list[Block] = []
    lines = text.splitlines()
    i = 0
    para: list[str] = []
    in_code = False
    code_block: Block | None = None

    def flush_para():
        if not para:
            return
        spans = _parse_inline(" ".join(para))
        block = Block("p")
        for wrapped in typeset.wrap(spans, width, size):
            block.lines.append(Line(wrapped, size=size))
        blocks.append(block)
        para.clear()

    while i < len(lines):
        raw = lines[i].rstrip()
        stripped = raw.strip()

        if stripped.startswith("```"):
            flush_para()
            if in_code and code_block is not None:
                blocks.append(code_block)
                code_block, in_code = None, False
            else:
                code_block, in_code = Block("code"), True
            i += 1
            continue
        if in_code and code_block is not None:
            code_block.lines.append(
                Line([Span(raw or " ")], size=size * 0.9, x=8,
                     color=(0.16, 0.18, 0.24), leading_mult=1.4, mono_all=True)
            )
            i += 1
            continue

        if not stripped:
            flush_para()
            i += 1
            continue

        heading = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if heading:
            flush_para()
            level = len(heading.group(1))
            title = heading.group(2).strip().strip("#").strip()
            mult = {1: 1.8, 2: 1.4, 3: 1.16}.get(level, 1.0)
            hsize = size * mult
            color = (0.05, 0.30, 0.42) if level <= 2 else (0.10, 0.12, 0.17)
            block = Block(f"h{level}", anchor=title, keep_with_next=True)
            for wrapped in typeset.wrap([Span(title, bold=True)], width, hsize):
                block.lines.append(Line(wrapped, size=hsize, color=color, leading_mult=1.32))
            blocks.append(block)
            i += 1
            continue

        if re.fullmatch(r"(-{3,}|\*{3,}|_{3,})", stripped):
            flush_para()
            blocks.append(Block("rule"))
            i += 1
            continue

        if stripped.startswith(">"):
            flush_para()
            block = Block("quote")
            for wrapped in typeset.wrap(
                _parse_inline(stripped.lstrip("> ").strip()), width - 18, size
            ):
                block.lines.append(
                    Line(wrapped, size=size, x=18, color=(0.30, 0.33, 0.40), leading_mult=1.45)
                )
            blocks.append(block)
            i += 1
            continue

        item = re.match(r"^(\s*)([-*+]|\d+[.)])\s+(.*)$", raw)
        if item and not stripped.startswith("|"):
            flush_para()
            depth = len(item.group(1)) // 2
            marker = item.group(2)
            bullet = "•" if marker in {"-", "*", "+"} else f"{marker.rstrip('.)')}."
            x = 4.0 + depth * 15.0
            spans = _parse_inline(item.group(3))
            wrapped_rows = typeset.wrap(spans, width - x - 20.0, size)
            for n, row in enumerate(wrapped_rows):
                lead = [Span(f"{bullet}  " if n == 0 else " " * (len(bullet) + 2))]
                blocks.append(
                    Block("li", lines=[Line(lead + row, size=size, x=x, leading_mult=1.5)])
                )
            i += 1
            continue

        if stripped.startswith("|") and stripped.count("|") >= 2:
            flush_para()
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                row_line = lines[i].strip().strip("|")
                if not TABLE_SEP_RE.match(lines[i].strip()):
                    rows.append([c.strip() for c in row_line.split("|")])
                i += 1
            tsize = size * 0.92
            colw, grid = _table_cells(rows, typeset, width, tsize)
            block = Block("table", columns=colw, cell_size=tsize)
            if grid:
                block.header_rows = [grid[0]]
                body = grid[1:]
            else:
                block.header_rows, body = [], grid
            block.rows = body
            block.lines = [Line([], size=tsize, leading_mult=1.4) for _ in body]
            blocks.append(block)
            continue

        para.append(stripped)
        i += 1

    flush_para()
    if in_code and code_block is not None and code_block.lines:
        blocks.append(code_block)
    return blocks


def _hex(value: str) -> tuple[float, float, float]:
    value = value.lstrip("#")
    if len(value) == 3:
        value = "".join(c * 2 for c in value)
    return tuple(int(value[j : j + 2], 16) / 255 for j in (0, 2, 4))  # type: ignore[return-value]


class PdfWriter:
    def __init__(self, args, typeset: Typesetter, ui: dict[str, str]):
        self.args, self.t, self.ui = args, typeset, ui
        self.W, self.H = PAGE_SIZES[args.page_size]
        self.ml = self.mr = args.margins
        self.mt, self.mb = args.margin_top, args.margins + 16
        self.accent = _hex(args.accent)
        self.body_w = self.W - self.ml - self.mr
        self.doc = fitz.open()
        self.page = None
        self.y = 0.0
        # One TextWriter per (page, colour): MuPDF embeds a font resource per writer, so a
        # writer per span would ship the same face hundreds of times over.
        self._writers: dict[tuple[float, float, float], Any] = {}
        self._dirty = False
        self.toc: list[list[Any]] = []

    # --- primitives -----------------------------------------------------------
    def flush(self):
        if self.page is not None and self._dirty:
            for color, tw in self._writers.items():
                tw.write_text(self.page, color=color)
        self._writers, self._dirty = {}, False

    def new_page(self, first: bool = False):
        self.flush()
        self.page = self.doc.new_page(width=self.W, height=self.H)
        self.y = self.mt
        if not first:
            self.page.draw_line(
                (self.ml, self.mt - 18), (self.W - self.mr, self.mt - 18),
                color=(0.86, 0.88, 0.91), width=0.5,
            )

    def draw(self, spans: list[Span], x: float, y: float, size: float, color, mono_all=False):
        tw = self._writers.setdefault(color, fitz.TextWriter(self.page.rect))
        for span in spans:
            if not span.text:
                continue
            span = Span(span.text, bold=span.bold, mono=span.mono or mono_all)
            face = self.t.face(span)
            tw.append((x, y), span.text, font=face, fontsize=size)
            if span.bold and self.t.faux_bold:
                tw.append((x + max(0.2, size * 0.02), y), span.text, font=face, fontsize=size)
            x += self.t.width(span, size)
        self._dirty = True
        return x

    def fits(self, need: float) -> bool:
        return self.y + need <= self.H - self.mb

    def room(self, need: float):
        if not self.fits(need):
            self.new_page()

    # --- block renderers ------------------------------------------------------
    def emit_line(self, line: Line):
        self.room(line.height)
        self.draw(line.spans, self.ml + line.x, self.y + line.size, line.size, line.color,
                  mono_all=line.mono_all)
        self.y += line.height

    def draw_table(self, block: Block):
        size = block.cell_size
        colw = block.columns
        xs, acc = [], self.ml
        for w in colw:
            xs.append(acc)
            acc += w + 12.0
        def row_heights(row) -> float:
            return max(len(c) for c in row) * size * 1.4 + 4.0

        def emit_grid_row(row, bold: bool):
            for r in range(max(len(c) for c in row)):
                self.room(size * 1.4)
                for c, cell_lines in enumerate(row):
                    if r < len(cell_lines):
                        self.draw(cell_lines[r], xs[c], self.y + size, size,
                                  (0.05, 0.28, 0.42) if bold else (0.10, 0.12, 0.17))
                self.y += size * 1.4
            self.page.draw_line((self.ml, self.y), (self.W - self.mr, self.y),
                                color=(0.88, 0.90, 0.93), width=0.4)
            self.y += 5.0

        for row in block.header_rows:
            emit_grid_row(row, bold=True)
        for row in block.rows:
            if not self.fits(row_heights(row)):
                self.new_page()
                for hdr in block.header_rows:
                    emit_grid_row(hdr, bold=True)
            emit_grid_row(row, bold=False)
        self.y += size * 0.5

    def draw_code(self, block: Block):
        top = self.y
        for line in block.lines:
            self.room(line.height)
            self.draw(line.spans, self.ml + line.x, self.y + line.size, line.size, line.color,
                      mono_all=True)
            self.y += line.height
        self.page.draw_rect(
            fitz.Rect(self.ml, top - 2, self.W - self.mr, self.y), color=None,
            fill=(0.96, 0.97, 0.985), overlay=False,
        )
        self.y += self.args.font_size * 0.6

    # --- document -------------------------------------------------------------
    def cover(self, title: str, meta: dict[str, str], rating: str, entries: list[tuple[int, str]]):
        a, ui = self.args, self.ui
        page = self.doc.new_page(width=self.W, height=self.H)
        self.page, self._writers, self._dirty = page, {}, False
        page.draw_rect(fitz.Rect(0, 0, self.W, self.H), color=None, fill=(0.985, 0.99, 0.995))
        page.draw_rect(fitz.Rect(0, 0, self.W, 5.5), color=None, fill=self.accent)
        size = a.font_size
        cy = self.H * 0.26
        self.y = cy
        self.draw([Span(ui["cover"])], self.ml, cy, size * 1.05, (0.38, 0.43, 0.49))
        cy += size * 2.6
        for row in self.t.wrap([Span(title, bold=True)], self.body_w, size * 2.15):
            self.draw(row, self.ml, cy, size * 2.15, (0.07, 0.09, 0.14))
            cy += size * 2.15 * 1.32
        page.draw_line((self.ml, cy + 4), (self.ml + 76, cy + 4), color=self.accent, width=2.4)
        cy += 30
        for key, value in meta.items():
            for row in self.t.wrap([Span(key), Span(ui["colon"] + value)], self.body_w, size)[:2]:
                self.draw(row, self.ml, cy, size, (0.30, 0.34, 0.40))
                cy += size * 1.8
        badge = fitz.Rect(self.ml, self.H * 0.70, min(self.ml + 210, self.W - self.mr),
                          self.H * 0.70 + 64)
        page.draw_rect(badge, color=None, fill=self.accent, radius=0.08)
        label = RATING_LABELS.get(rating, rating) if a.rating_labels else rating
        self.draw([Span(f"{ui['rating']}{ui['colon']}{label}", bold=True)],
                  badge.x0 + 14, badge.y0 + 28, size * 1.3, (1, 1, 1))
        self.draw([Span(f"{rating}   {meta.get('Trade date') or meta.get('trade date') or ''}")],
                  badge.x0 + 14, badge.y0 + 50, size * 0.86, (0.90, 0.95, 1.0))
        if a.toc and entries:
            ty = badge.y1 + 34
            self.draw([Span(ui["sections"], bold=True)], self.ml, ty, size * 1.05,
                      (0.35, 0.40, 0.46))
            ty += size * 2.1
            for _level, entry in [e for e in entries if e[0] == 1][: a.toc_items]:
                if ty > self.H - self.mb - 12:
                    break
                self.draw(self.t.wrap([Span(entry)], self.body_w, size * 0.94)[0],
                          self.ml + 3, ty, size * 0.94, (0.24, 0.28, 0.34))
                ty += size * 1.55
        self.flush()

    def footer(self):
        total = self.doc.page_count
        a, ui = self.args, self.ui
        for pno in range(total):
            if pno == 0 or not a.footer:
                continue
            page = self.doc[pno]
            self._writers, self._dirty = {}, False
            left = a.footer_text or self.footer_label
            if left:
                self.page = page
                self.draw([Span(left)], self.ml, self.H - self.mb + 26, a.font_size * 0.78,
                          (0.54, 0.58, 0.64))
            label = (ui["page"] % (pno + 1)) + " " + (ui["of"] % total)
            w = self.t.regular.text_length(label, fontsize=a.font_size * 0.78)
            self.page = page
            self.draw([Span(label)], self.W - self.mr - w, self.H - self.mb + 26,
                      a.font_size * 0.78, (0.54, 0.58, 0.64))
            self.flush()

    def run(self, md_path: Path, out_path: Path) -> dict[str, Any]:
        text = md_path.read_text(encoding="utf-8")
        a = self.args
        size = a.font_size
        blocks = parse_markdown(text, self.t, self.body_w, size)

        title = a.title or md_path.stem
        meta: dict[str, str] = {}
        for ln in text.splitlines()[:14]:
            s = ln.strip()
            if s.startswith("# "):
                title = s[2:].strip()
            elif ":" in s and not s.startswith(("#", "-", "|", ">", "*")) and len(s) < 200:
                key, value = s.split(":", 1)
                if value.strip() and len(key.strip()) <= 26 and not value.strip().startswith("**"):
                    meta[key.strip()] = value.strip()
        for extra in a.meta or []:
            key, _, value = extra.partition("=")
            if key.strip():
                meta[key.strip()] = value.strip() or key.strip()
        # The report tree concatenates every agent's prose, and the bull/bear debate uses
        # rating words freely ("...Sell on any rally..."), so scanning the whole document can
        # read the wrong agent's word. Scope to the PM section, then fall back.
        pm_section = text[text.find("## V. Portfolio Manager"):] if "## V. Portfolio Manager" in text else ""
        rating = a.rating or extract_rating(pm_section or text) or "REVIEW"
        self.footer_label = " · ".join(
            v for v in (meta.get("Asset") or a.title, meta.get("Trade date") or a.date) if v
        ) or md_path.stem

        entries = [
            (1 if b.kind in {"h1", "h2"} else 2, b.anchor)
            for b in blocks
            if b.anchor and b.kind in {"h1", "h2", "h3"} and b.anchor != title
        ]
        if not a.no_cover:
            self.cover(title, meta, rating, entries)

        self.new_page(first=True)
        for block in blocks:
            if block.kind == "rule":
                self.room(size)
                self.page.draw_line((self.ml, self.y), (self.W - self.mr, self.y),
                                    color=(0.86, 0.88, 0.91), width=0.6)
                self.y += size * 0.8
                continue
            if block.kind == "table":
                self.room(size * 4)
                self.draw_table(block)
                continue
            if block.kind == "code":
                self.draw_code(block)
                continue
            if block.keep_with_next:
                self.room(size * 1.4 + size * 1.5)
                if block.anchor and block.anchor != title:
                    self.toc.append([1 if block.kind in {"h1", "h2"} else 2, block.anchor,
                                     self.doc.page_count])
            for line in block.lines:
                self.emit_line(line)
            self.y += size * (0.62 if block.kind.startswith("h") else 0.34)

        self.flush()
        self.footer()
        if a.toc and self.toc:
            with contextlib.suppress(Exception):
                self.doc.set_toc(self.toc)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if not a.no_subset:
            with contextlib.suppress(Exception):
                self.doc.subset_fonts()
        self.doc.save(out_path, garbage=4, deflate=True, clean=True)
        info = {
            "pages": self.doc.page_count,
            "rating": rating,
            "bookmarks": len(self.toc),
            "bytes": out_path.stat().st_size,
            "title": title,
        }
        self.doc.close()
        return info


def detect_lang(text: str) -> str:
    """Pick the script family from the report itself, so ``--lang auto`` needs no hints.

    Han counts alone cannot tell Chinese from Japanese, so kana decide the tie; a report
    with no CJK at all keeps English cover strings while still using the symbol-complete face.
    """
    sample = text[:20000]
    han = sum(1 for c in sample if "\u4e00" <= c <= "\u9fff")
    kana = sum(1 for c in sample if "\u3040" <= c <= "\u30ff")
    hangul = sum(1 for c in sample if "\uac00" <= c <= "\ud7a3")
    if kana and kana * 3 > han:
        return "ja"
    if hangul and hangul > han:
        return "ko"
    return "zh" if han > 20 else "en"


def build_typesetter(args, face: str | None = None) -> tuple[Typesetter, str]:
    if args.font:
        regular = fitz.Font(fontfile=str(Path(args.font).expanduser()))
        bold = (fitz.Font(fontfile=str(Path(args.font_bold).expanduser()))
                 if args.font_bold else regular)
        faux_bold = not args.font_bold
    else:
        regular = fitz.Font(face or BUILTIN_FACES.get(getattr(args, "lang", "en"), "china-t"))
        bold, faux_bold = regular, True
    try:
        mono = fitz.Font("cour")
    except Exception:
        mono = None
    missing = [c for c in "中文測試火箭" if not regular.has_glyph(ord(c))]
    if missing:
        print(
            f"warning: font '{regular.name}' has no glyphs for {''.join(missing)}; "
            "pass --font with a CJK family (e.g. Noto Sans TC) for correct output",
            file=sys.stderr,
        )
    return Typesetter(regular, bold, mono, faux_bold), regular.name


RENDER_DEFAULTS: dict[str, Any] = {
    "lang": "auto",
    "page_size": "a4",
    "font_size": 9.6,
    "margins": 46.0,
    "margin_top": 60.0,
    "font": None,
    "font_bold": None,
    "accent": "#0d5c73",
    "title": None,
    "rating": None,
    "meta": None,
    "date": None,
    "no_cover": False,
    "toc": True,
    "toc_items": 12,
    "footer": True,
    "footer_text": None,
    "rating_labels": True,
    "no_subset": False,
    "dpi_preview": None,
}


def render_pdf(report: Path | str, out: Path | str | None = None, **overrides) -> dict[str, Any]:
    """Typeset ``report`` to a PDF and return ``{pages, bookmarks, rating, bytes, path, font}``.

    Programmatic entry point (used by ``scripts/agent_supplied_run.py --pdf``); unknown
    overrides raise so a typo cannot silently fall back to a default.
    """
    unknown = set(overrides) - set(RENDER_DEFAULTS)
    if unknown:
        raise TypeError(f"unknown render option(s): {', '.join(sorted(unknown))}")
    args = argparse.Namespace(**{**RENDER_DEFAULTS, **overrides, "report": Path(report)})
    path = Path(report)
    if args.lang == "auto":
        args.lang = detect_lang(path.read_text(encoding="utf-8"))
    typeset, font_name = build_typesetter(args)
    out_path = Path(out) if out else path.with_suffix(".pdf")
    info = PdfWriter(args, typeset, UI.get(args.lang, UI["en"])).run(path, out_path)
    info.update(font=font_name, path=out_path, lang=args.lang)
    return info


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Typeset a TradingAgents markdown report as a PDF (CJK- and symbol-safe).",
    )
    ap.add_argument("report", type=Path, nargs="?", help="path to complete_report.md")
    ap.add_argument("--out", type=Path, help="output PDF (default: alongside the report)")
    ap.add_argument("--lang", default="auto", choices=["auto", *sorted(UI)],
                    help="language of the renderer's own cover/footer strings (default: detect)")
    ap.add_argument("--page-size", default="a4", choices=sorted(PAGE_SIZES))
    ap.add_argument("--font-size", type=float, default=9.6)
    ap.add_argument("--margins", type=float, default=46.0, help="left/right/bottom margin, pt")
    ap.add_argument("--margin-top", type=float, default=60.0)
    ap.add_argument("--font", help="body face: .ttf/.otf/.ttc path (e.g. Noto Sans TC)")
    ap.add_argument("--font-bold", help="true bold face for --font (else faux-bold is drawn)")
    ap.add_argument("--accent", default="#0d5c73", help="accent colour: rules, badge")
    ap.add_argument("--title", help="cover title override")
    ap.add_argument("--rating", help="override the rating word on the cover badge")
    ap.add_argument("--meta", action="append", metavar="KEY=VALUE",
                    help="extra cover metadata line (repeatable); merged after the parsed header")
    ap.add_argument("--date", help="trade date for the footer when the report omits it")
    ap.add_argument("--no-cover", action="store_true")
    ap.add_argument("--no-toc", dest="toc", action="store_false", help="skip PDF outline/bookmarks")
    ap.add_argument("--toc-items", type=int, default=12, help="contents entries on the cover")
    ap.add_argument("--no-footer", dest="footer", action="store_false")
    ap.add_argument("--footer-text", help="override the running footer's left side")
    ap.add_argument("--no-rating-labels", dest="rating_labels", action="store_false",
                    help="print the raw English rating word instead of a translated label")
    ap.add_argument("--no-subset", action="store_true", help="skip font subsetting (larger file)")
    ap.add_argument("--dpi-preview", type=int,
                    help="also rasterise the first, third and last page to PNG next to the PDF")
    ap.set_defaults(toc=True, footer=True, rating_labels=True)
    args = ap.parse_args(argv)

    if args.report is None:
        ap.error("a report path is required")
    if not args.report.exists():
        print(f"error: no such report: {args.report}", file=sys.stderr)
        return 2
    options = {k: v for k, v in vars(args).items() if k in RENDER_DEFAULTS and k != "report"}
    info = render_pdf(args.report, args.out, **options)
    print(
        f"pdf: {info['path']}\n"
        f"  {info['pages']} pages · {info['bookmarks']} bookmarks · lang {info['lang']} · "
        f"font {info['font']} · rating {info['rating']} · {info['bytes'] / 1024:.0f} KB"
    )
    if args.dpi_preview:
        doc = fitz.open(info["path"])
        for pno in sorted({0, min(2, doc.page_count - 1), doc.page_count - 1}):
            doc[pno].get_pixmap(dpi=args.dpi_preview).save(
                Path(info["path"]).with_name(f"{Path(info['path']).stem}.p{pno + 1}.png")
            )
        doc.close()
        print(f"  previews written for pages 1, 3 and {info['pages']}")
    return 0



if __name__ == "__main__":
    raise SystemExit(main())
