#!/usr/bin/env python3
"""Pair a report with its translation into one bilingual document, English first.

``write_report_tree`` writes English-by-default (unless ``TRADINGAGENTS_OUTPUT_LANGUAGE``
localises the run), so a 中英對照 deliverable needs the two texts interleaved before
typesetting. Pairing is positional, not fuzzy: both documents come from the same framework
section order (``## I. Analyst Team Reports`` → four ``### `` analysts, and so on), so
sections and subsections are matched by index and a count mismatch is a hard error rather
than a silently mis-paired report.

Layout per subsection::

    ### Market Analyst ｜ 市場（技術面）分析師

    #### English
    …original…

    #### 繁體中文
    …translation…

The analysts' own ``#``/``##`` headings are folded to ``#####`` so the PDF outline stays at
the framework's three levels (report → section → agent) and the language labels stay visible
at a glance. Trailing sections present only in the translation (an appendix, say) are kept.

Usage
-----
    python scripts/merge_bilingual_report.py complete_report.md complete_report_zh.md \
        --out complete_report_bilingual.md --title "Trading Analysis: RKLB ｜ 交易分析報告（中英對照）"
    python scripts/report_to_pdf.py complete_report_bilingual.md   # then typeset it

Exit status: 0 on success, 1 on a pairing error, 2 on a usage/IO error.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

SECTION_RE = re.compile(r"^##\s+((?:[IVX]+\.|Appendix|附錄)[^\n]*)$")
SUBSECTION_RE = re.compile(r"^###\s+(.*)$")
HEAD_RE = re.compile(r"^(#{1,4})\s+(.*)$")


def split_sections(text: str) -> tuple[list[str], list[tuple[str, list[tuple[str, list[str]]]]]]:
    """Return (preamble lines, [(section heading, [(subsection heading, body lines), …]), …])."""
    preamble: list[str] = []
    sections: list[tuple[str, list[tuple[str, list[str]]]]] = []
    pending: tuple[str, list[str]] | None = None

    def close_pending() -> None:
        nonlocal pending
        if pending is not None:
            title, body = pending
            # Blank prose between "## Section" and the first "### Sub" is not a subsection.
            if title or any(line.strip() for line in body):
                sections[-1][1].append(pending)
            pending = None

    for line in text.splitlines():
        section = SECTION_RE.match(line)
        sub = SUBSECTION_RE.match(line)
        if section:
            close_pending()
            sections.append((section.group(1).strip(), []))
            continue
        if sub and sections:
            close_pending()
            pending = (sub.group(1).strip(), [])
            continue
        if not sections:  # title + metadata before the first "## "
            preamble.append(line)
            continue
        if pending is None:
            # Section prose with no subsection heading: keep it under an empty title.
            pending = ("", [])
        pending[1].append(line)
    close_pending()
    return preamble, sections


def demote(body: list[str]) -> list[str]:
    """Fold an agent's own headings into ``#####`` labels that stay out of the PDF outline.

    A leading ``#`` title ("# Technical Market Report — RKLB") repeats the merged subsection
    heading, so level-1 headings are dropped; ``##``–``####`` become level-5 labels. The
    result is idempotent: an already-demoted ``#####`` line is left alone.
    """
    out: list[str] = []
    for raw in body:
        line = raw.rstrip()
        match = HEAD_RE.match(line)
        if match is None:
            out.append(line)
        elif len(match.group(1)) == 1:
            continue
        else:
            out.append(f"##### {match.group(2).strip()}")
    return out


NUM_PREFIX_RE = re.compile(r"^[IVX]+\.\s*")


def join(en: str, zh: str, sep: str) -> str:
    """Join an English heading with its translation, e.g. “Market Analyst ｜ 市場分析師”."""
    en, zh = en.strip(), zh.strip()
    if NUM_PREFIX_RE.match(en) and NUM_PREFIX_RE.match(zh):
        zh = NUM_PREFIX_RE.sub("", zh)  # "## I. x ｜ I. y" would repeat the numeral
    if not zh or zh == en:
        return en
    return f"{en}{sep}{zh}" if en else zh


def merge(en_text: str, zh_text: str, *, title: str | None, sep: str, en_label: str,
          zh_label: str) -> tuple[str, dict[str, Any]]:
    """Interleave two aligned reports; raise ValueError when they do not line up."""
    en_pre, en_sections = split_sections(en_text)
    zh_pre, zh_sections = split_sections(zh_text)
    if len(zh_sections) < len(en_sections):
        raise ValueError(
            f"translation has {len(zh_sections)} sections, original has {len(en_sections)}"
        )
    pairs = list(zip(en_sections, zh_sections[: len(en_sections)], strict=True))

    out: list[str] = []
    if title:
        out += [f"# {title}", ""]
        keys = set()
        for line in en_pre + zh_pre:
            if HEAD_RE.match(line.strip()) or not line.strip():
                continue
            # Cover metadata ("Asset: …", "Trade date: …") usually lives in the translation's
            # preamble, so merge both sides, English first, without duplicating a key.
            key = line.split(":", 1)[0].strip()
            if ":" in line and 0 < len(key) <= 26:
                if key in keys:
                    continue
                keys.add(key)
            out.append(line)
        out.append("")
    else:
        out += en_pre

    paired = skipped = appended = 0
    for en_sec, zh_sec in pairs:
        (en_head, en_subs), (zh_head, zh_subs) = en_sec, zh_sec
        out += [f"## {join(en_head, zh_head, sep)}", ""]
        if len(zh_subs) < len(en_subs):
            raise ValueError(
                f"section {en_head!r}: {len(en_subs)} subsections vs {len(zh_subs)} translated"
            )
        for (e_title, e_body), (z_title, z_body) in zip(en_subs, zh_subs[: len(en_subs)], strict=True):
            out += [f"### {join(e_title, z_title, sep)}", "", f"#### {en_label}", ""]
            out += demote(e_body)
            out += ["", f"#### {zh_label}", ""]
            out += [ln for ln in (line.rstrip() for line in z_body) if ln]
            out += ["", "---", ""]
            paired += 1
        skipped += max(len(zh_subs) - len(en_subs), 0)

    for extra_head, extra_subs in zh_sections[len(en_sections) :]:
        appended += 1
        out += [f"## {extra_head}", ""]
        for sub_title, sub_body in extra_subs:
            if sub_title:
                out += [f"### {sub_title}", ""]
            out += [ln for ln in (line.rstrip() for line in sub_body) if ln]
            out.append("")

    text = "\n".join(out).rstrip() + "\n"
    return text, {
        "sections": len(pairs),
        "subsections": paired,
        "extra_translation_blocks": skipped,
        "translation_only_sections": appended,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Interleave an English report and its translation into one bilingual document.",
    )
    ap.add_argument("original", type=Path, help="the run's complete_report.md")
    ap.add_argument("translation", type=Path, help="the translated report markdown")
    ap.add_argument("--out", type=Path, help="output path (default: <original stem>_bilingual.md)")
    ap.add_argument("--title", help="cover/H1 title override")
    ap.add_argument("--sep", default=" ｜ ", help="separator between an English and Chinese heading")
    ap.add_argument("--en-label", default="English", help="language label above the original text")
    ap.add_argument("--zh-label", default="繁體中文", help="language label above the translation")
    args = ap.parse_args(argv)

    try:
        en_text = args.original.read_text(encoding="utf-8")
        zh_text = args.translation.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"error: cannot read inputs: {exc}", file=sys.stderr)
        return 2
    out_path = args.out or args.original.with_name(f"{args.original.stem}_bilingual.md")
    try:
        text, stats = merge(
            en_text, zh_text, title=args.title, sep=args.sep,
            en_label=args.en_label, zh_label=args.zh_label,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    out_path.write_text(text, encoding="utf-8")
    print(
        f"wrote {out_path}\n  {stats['sections']} sections · {stats['subsections']} paired "
        f"subsections · {stats['translation_only_sections']} translation-only section(s) appended"
        f" · {stats['extra_translation_blocks']} extra translated block(s)"
    )
    print(f"next: python scripts/report_to_pdf.py {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
