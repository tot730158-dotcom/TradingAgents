# 報告 PDF 書架 · report PDF library

Every finished TradingAgents report PDF that is worth keeping lives here. This folder is
**tracked in Git on purpose** (the rest of `reports/` is ignored), so a sandbox reset that
clears `reports/` cannot clear the deliverables — they come back with the branch.

Default cut per ticker: the 純繁體中文 PDF. Re-file or add with:

```bash
python scripts/pdf_library.py sync            # every reports/<TICKER>_<DATE>/ not yet shelved
python scripts/pdf_library.py add <pdf> --ticker OKLO --date 2026-09-15 --markdown <zh.md>
python scripts/pdf_library.py build           # regenerate this table and index.html
```

| 日期 | 股票 | 語言 | 決策 | 頁 | 檔案 |
|---|---|---|---|---|---|
| 2026-09-15 | OKLO | 繁體中文 | Underweight 減碼 | 10 | [`2026-09-15_OKLO_traditional_chinese.pdf`](2026-09-15_OKLO_traditional_chinese.pdf) · [markdown](2026-09-15_OKLO_traditional_chinese.md) |

`manifest.json` holds the machine-readable version of this table.
研究用，不構成投資建議。
