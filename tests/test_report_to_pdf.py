"""Markdown report tree → typeset PDF, using PyMuPDF alone.

Each test here maps to a failure a real report produced: the base-14 faces raise on the
typographic symbols analyst prose is full of (≈ → ± —) instead of substituting; a non-Latin
run cannot be typeset without an embeddable CJK face; a ``TextWriter`` per text span shipped
the same font ~1,100 times (4 MB for 17 pages); and the cover badge read the *bull/bear
debate*'s rating words rather than the Portfolio Manager's decision, because the concatenated
report contains every agent's prose.
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import pytest

pymupdf = pytest.importorskip("pymupdf")

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "report_to_pdf.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("report_to_pdf", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    # Register before exec: the module's @dataclass classes resolve their annotations
    # through sys.modules[cls.__module__], which is unset for a bare spec-loaded module.
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    return module


# The renderer is a script, not a package module: import it by path once.
rtp = _load_module()

REPORT = """# Trading Analysis: RKLB

Generated: 2026-09-15

Asset: Rocket Lab Corporation (NASDAQ: RKLB)

Trade date: 2026-09-15

## I. Analyst Team Reports

### Market Analyst

Price sits **below** the $67.50 collar floor (≈ -8% spot → no float conversion).

| level | price | why |
|---|---|---|
| support | $58.25 | August low |
| trigger | $67.50 | collar floor |

- RSI(14) ≈ 33 — no confirmed divergence
- ATR $3.82 (6.1%) ± wide prints

> caveat: the retail band is inferred, not counted.

```
python scripts/agent_supplied_run.py pack.json --validate-only
```

## V. Portfolio Manager Decision

### Portfolio Manager

**Rating**: Hold

Target price 80.00 with a protective line at 58.00.

## V. Portfolio Manager — bull/bear noise

The bear researcher argued to Sell on any rally toward $71; the bull said Buy below $60.
"""

ZH_REPORT = """# 交易分析報告：RKLB（Rocket Lab 火箭實驗室）

Generated: 2026-09-15

Trade date: 2026-09-15

## I. 分析師團隊報告

### 市場分析師

股價 $62.55 低於 $67.50 的 collar 下限（約 -8%），因此換股比例凍結在 0.4000，稀釋上限約
4,240 萬股；每再跌 $1，銥星股東對價減少約 $0.40。

**結論：持有** —— 營運改善、價格不便宜、獲利遞延。

## V. 組合經理最終決策

### 組合經理

**Rating**: Hold
"""


def _pdf(tmp_path, text, **overrides):
    md = tmp_path / "complete_report.md"
    md.write_text(text, encoding="utf-8")
    return rtp.render_pdf(md, tmp_path / "out.pdf", **overrides)


@pytest.mark.unit
def test_markdown_maps_to_block_kinds(tmp_path):
    typeset, _ = rtp.build_typesetter(argparse.Namespace(font=None, font_bold=None, lang="en"))
    md = tmp_path / "complete_report.md"
    md.write_text(REPORT, encoding="utf-8")
    blocks = rtp.parse_markdown(md.read_text(encoding="utf-8"), typeset, 500.0, 9.6)
    kinds = [b.kind for b in blocks]
    assert kinds[0] == "h1"  # document title, excluded from the outline later
    assert "table" in kinds and "code" in kinds and "quote" in kinds and "rule" not in kinds
    assert "li" in kinds
    table = next(b for b in blocks if b.kind == "table")
    # Header row is tracked apart from the body so it can repeat across a page break.
    assert len(table.header_rows) == 1 and len(table.rows) == 2
    assert len(table.columns) == 3


@pytest.mark.unit
def test_symbols_and_bold_render_without_base14_crash(tmp_path):
    info = _pdf(tmp_path, REPORT, lang="en")
    assert info["pages"] >= 2
    with pymupdf.open(info["path"]) as doc:
        text = "\n".join(page.get_text() for page in doc)
    # The characters that made fz_show_string raise when body text used helv/cour.
    for token in ("≈", "→", "±", "—", "$67.50"):
        assert token in text
    assert "## " not in text and "**" not in text  # markdown markers are consumed, not shown
    with pymupdf.open(info["path"]) as doc:
        assert doc.get_toc()  # outline written


@pytest.mark.unit
def test_cover_badge_reads_the_portfolio_manager_not_the_debate(tmp_path):
    info = _pdf(tmp_path, REPORT, lang="en")
    assert info["rating"] == "Hold"
    with pymupdf.open(info["path"]) as doc:
        cover = doc[0].get_text()
    assert "Final decision" in cover and "Hold" in cover
    assert "Sell" not in cover  # the bear's "Sell on any rally" must not reach the badge
    override = rtp.render_pdf(
        Path(info["path"]).with_name("complete_report.md"), None, lang="en", rating="Underweight"
    )
    assert override["rating"] == "Underweight"


@pytest.mark.unit
def test_chinese_report_typesets_and_detects_itself(tmp_path):
    info = _pdf(tmp_path, ZH_REPORT)  # lang defaults to "auto"
    assert info["lang"] == "zh"
    with pymupdf.open(info["path"]) as doc:
        text = "\n".join(page.get_text() for page in doc)
        cover = doc[0].get_text()
        assert doc.page_count >= 2
        # Extraction round-trips the Han codepoints, so glyphs are really embedded — and
        # Traditional forms must not be substituted by their Simplified counterparts.
        assert "火箭實驗室" in text and "換股比例" in text
        assert "国" not in text and "关" not in text
    assert "最終決策" in cover and "持有" in cover  # translated chrome, no --lang needed
    assert "第" in text  # CJK footer


@pytest.mark.unit
def test_font_duplication_and_size_regression(tmp_path):
    info = _pdf(tmp_path, "\n\n".join([REPORT] * 4), lang="en")
    # ~1,100 embedded font copies used to push a 17-page report to 4 MB and 48 s.
    assert info["bytes"] < 600_000
    with pymupdf.open(info["path"]) as doc:
        xrefs = {xref[0] for page in doc for xref in page.get_fonts()}
        assert len(xrefs) <= 4
        assert sum(len(page.get_text()) for page in doc) > 4000


@pytest.mark.unit
def test_cli_options_and_unknown_option(tmp_path):
    md = tmp_path / "complete_report.md"
    md.write_text(ZH_REPORT, encoding="utf-8")
    code = rtp.main([str(md), "--out", str(tmp_path / "cli.pdf"), "--no-cover", "--dpi-preview", "70"])
    assert code == 0
    with pymupdf.open(tmp_path / "cli.pdf") as doc:
        # --no-cover starts on body text, so the title heading is on page 1.
        assert "市場分析師" in doc[0].get_text()
        assert doc.get_toc()
    assert (tmp_path / "cli.p1.png").exists()
    with pytest.raises(TypeError, match="unknown render option"):
        rtp.render_pdf(md, page_size="a4", pagse_size="letter")


@pytest.mark.unit
def test_missing_report_is_a_usage_error(tmp_path):
    assert rtp.main([str(tmp_path / "nope.md")]) == 2


@pytest.mark.unit
def test_empty_report_still_produces_a_document(tmp_path):
    md = tmp_path / "complete_report.md"
    md.write_text("", encoding="utf-8")
    info = rtp.render_pdf(md)
    assert info["pages"] >= 1
    assert info["rating"] == "REVIEW"
    assert pymupdf.open(info["path"]).page_count == info["pages"]

@pytest.mark.unit
def test_outline_stops_at_the_agent_level(tmp_path):
    md = tmp_path / "complete_report.md"
    md.write_text(
        ZH_REPORT + "\n#### English\n\nlabelled block\n\n##### deep\n\nmore\n", encoding="utf-8"
    )
    info = rtp.render_pdf(md, tmp_path / "out.pdf", lang="zh")
    with pymupdf.open(info["path"]) as doc:
        titles = [entry[1] for entry in doc.get_toc()]
    assert "English" not in " ".join(titles) and "deep" not in " ".join(titles)
    assert any("組合經理" in t for t in titles)  # h3 agents are still bookmarked


@pytest.mark.unit
def test_html_reader_is_self_contained_and_dark_mode_ready(tmp_path):
    info = _pdf(tmp_path, ZH_REPORT, lang="zh")
    # The PDF carries the report title, so the reader header is not just a filename.
    with pymupdf.open(info["path"]) as doc:
        assert doc.metadata["title"] == "交易分析報告：RKLB（Rocket Lab 火箭實驗室）"
        pages = doc.page_count
    reader = rtp.write_html_reader(info["path"], tmp_path / "read.html", lang="zh")
    assert reader["pages"] == pages
    html = (tmp_path / "read.html").read_text(encoding="utf-8")
    assert html.count("data:image/jpeg;base64,") == pages  # every page inlined
    assert "src=\"http" not in html and "href=\"http" not in html  # nothing to fetch
    assert 'lang="zh-Hant"' in html and "@media print" in html and "color-scheme: light dark" in html
    assert 'id="theme-dark"' in html and 'id="invert"' in html and 'onclick="window.print()"' in html
