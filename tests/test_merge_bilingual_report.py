"""Bilingual (English + 繁體中文) report merging, positional by framework section order.

``write_report_tree`` emits ``## I. Analyst Team Reports`` with four ``### `` analysts, then
three research subsections, one trader, three risk voices and one PM. A translation of that
document has the same shape, so pairing is done by index — and a mismatch must fail loudly,
because a silently shifted pairing puts the bear case under the bull heading.
"""

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "merge_bilingual_report.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("merge_bilingual_report", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


merge = pytest.fixture(scope="module")(_load_module)

EN = """# Trading Analysis Report: RKLB

Generated: 2026-09-15

## I. Analyst Team Reports

### Market Analyst
# Technical Market Report — RKLB
## Trend structure
Price is below every moving average.

### Sentiment Analyst
## Source-by-source breakdown
Mixed.

## II. Research Team Decision

### Bull Researcher
Growth is compounding.
"""

ZH = """# 交易分析報告：RKLB

## I. 分析師團隊報告

### 市場分析師
股價位於所有均線之下。

### 情緒分析師
混合。

## II. 研究團隊決議

### 多方研究員
營收持續複合成長。

## 附錄：方法論
僅供研究使用。
"""


def test_pairs_by_position_and_labels_each_language(merge):
    text, stats = merge.merge(EN, ZH, title="T ｜ 標題", sep=" ｜ ", en_label="English",
                              zh_label="繁體中文")
    assert stats == {
        "sections": 2,
        "subsections": 3,
        "extra_translation_blocks": 0,
        "translation_only_sections": 1,
    }
    assert text.startswith("# T ｜ 標題")
    assert "Generated: 2026-09-15" in text  # original preamble metadata survives
    # The translation's own cover line is merged in, so the PDF cover/footer can use it.
    zh_with_meta = ZH.replace("# 交易分析報告：RKLB\n", "# 交易分析報告：RKLB\n\nTrade date: 2026-09-15\n")
    merged, _ = merge.merge(EN, zh_with_meta, title="T", sep=" ｜ ", en_label="English",
                            zh_label="繁體中文")
    assert "Trade date: 2026-09-15" in merged
    assert "## I. Analyst Team Reports ｜ 分析師團隊報告" in text  # numeral not doubled
    assert "### Market Analyst ｜ 市場分析師" in text
    assert "#### English" in text and "#### 繁體中文" in text
    assert "##### Trend structure" in text  # the agent's own heading is folded to a label
    assert "# Technical Market Report — RKLB" not in text  # its title repeats the section
    assert "股價位於所有均線之下。" in text
    assert "## 附錄：方法論" in text  # translation-only section appended, not dropped
    # English stays ahead of its translation inside every subsection.
    assert text.index("Price is below") < text.index("股價位於所有均線之下")


def test_blank_prose_between_section_and_first_subsection_is_not_a_subsection(merge):
    text, _ = merge.merge(EN, ZH, title=None, sep=" ｜ ", en_label="English", zh_label="繁體中文")
    assert "### \n" not in text and "\n### \n\n" not in text
    # Without a --title the original H1 and its metadata lines are kept verbatim.
    assert text.startswith("# Trading Analysis Report: RKLB")


def test_missing_translation_for_one_subsection_is_an_error(merge):
    truncated = ZH.replace("### 情緒分析師\n混合。\n\n", "")
    with pytest.raises(ValueError, match="subsections vs"):
        merge.merge(EN, truncated, title=None, sep=" ｜ ", en_label="English",
                    zh_label="繁體中文")


def test_fewer_translated_sections_than_original_is_an_error(merge):
    short = ZH.split("## II.")[0]
    with pytest.raises(ValueError, match="sections, original"):
        merge.merge(EN, short, title=None, sep=" ｜ ", en_label="English", zh_label="繁體中文")


def test_cli_writes_next_to_the_original(tmp_path, merge):
    en = tmp_path / "complete_report.md"
    zh = tmp_path / "complete_report_zh.md"
    en.write_text(EN, encoding="utf-8")
    zh.write_text(ZH, encoding="utf-8")
    assert merge.main([str(en), str(zh)]) == 0
    out = tmp_path / "complete_report_bilingual.md"
    assert out.exists() and "#### 繁體中文" in out.read_text(encoding="utf-8")
    assert merge.main([str(en), str(tmp_path / "missing.md")]) == 2
