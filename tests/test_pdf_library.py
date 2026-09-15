"""The Git-tracked PDF shelf (``reports/library/``).

The failure this tool exists for is a workspace reset: it clears ``reports/`` (untracked) and
``~/.tradingagents``, and a run whose sections were authored by the agent cannot be replayed.
So the tests here are mostly about the shelf being *idempotent and self-describing* — re-filing
the same bytes must not duplicate an entry, a PDF dropped in by hand must still get an entry,
and a deleted file must not leave a dead card on the page.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

pymupdf = pytest.importorskip("pymupdf")

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "pdf_library.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("pdf_library", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    # Register before exec: Entry's @dataclass resolves its annotations through
    # sys.modules[cls.__module__], which a bare spec-load leaves unset.
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    return module


lib = pytest.fixture(scope="module")(_load_module)


def make_pdf(path: Path, *, title: str = "", rating: str = "", text: str = "Hello") -> Path:
    """A real one-page PDF: the shelf reads pages/title/rating out of the document info."""
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 96), text)
    doc.set_metadata({"title": title, "keywords": f"rating: {rating}" if rating else ""})
    doc.save(path)
    doc.close()
    return path


@pytest.fixture
def run_dir(tmp_path):
    """A ``reports/OKLO_2026-09-15``-shaped run: PDF, reader, markdown, pack.json."""
    root = tmp_path / "reports" / "OKLO_2026-09-15"
    root.mkdir(parents=True)
    pdf = make_pdf(root / "OKLO_2026-09-15_繁體中文報告.pdf",
                   title="交易分析報告：OKLO", rating="Underweight")
    pdf.with_suffix(".html").write_text("<html></html>", encoding="utf-8")
    (root / "complete_report_zh.md").write_text(
        "# 交易分析報告：OKLO\n\n## V. 組合經理最終決策\n\n**Rating**: Underweight\n", encoding="utf-8"
    )
    (root / "pack.json").write_text(json.dumps({
        "ticker": "OKLO", "trade_date": "2026-09-15",
        "instrument_context": "Oklo Inc. (NYSE: OKLO) — advanced fission developer",
    }, ensure_ascii=False), encoding="utf-8")
    return root


def shelf(tmp_path) -> Path:
    return tmp_path / "reports" / "library"


# ---------------------------------------------------------------------------------------------
# naming and validation
# ---------------------------------------------------------------------------------------------


def test_slug_is_deterministic_per_language(lib):
    assert lib.derive_slug(ticker="OKLO", date_str="2026-09-15", lang="zh") == \
        "2026-09-15_OKLO_traditional_chinese"
    assert lib.derive_slug(ticker="OKLO", date_str="2026-09-15", lang="bilingual") == \
        "2026-09-15_OKLO_bilingual_en_zh"


def test_rejects_a_non_pdf_and_a_bad_ticker_or_date(lib, tmp_path, run_dir):
    source = run_dir / "OKLO_2026-09-15_繁體中文報告.pdf"
    fake = tmp_path / "not_a_report.pdf"
    fake.write_bytes(b"%PDF is love")
    with pytest.raises(lib.LibraryError, match="not a PDF"):
        lib.Library(shelf(tmp_path)).add(fake, ticker="OKLO", date_str="2026-09-15")

    base = {"ticker": "OKLO", "date_str": "2026-09-15"}
    bad_cases = [{"ticker": "OK LO"}, {"ticker": ""}, {"ticker": "OKLO**"},
                 {"date_str": "2026-9-1"}, {"date_str": "yesterday"}, {"lang": "de"},
                 {"slug": "bad/slug"}]
    for bad in bad_cases:
        with pytest.raises(lib.LibraryError, match="bad |unknown language"):
            lib.Library(shelf(tmp_path)).add(source, **{**base, **bad})
    assert not shelf(tmp_path).exists()  # a rejected add leaves no half-written shelf


def test_oversized_pdf_is_refused_before_it_bloats_git(lib, tmp_path, run_dir, monkeypatch):
    monkeypatch.setattr(lib, "MAX_PDF_BYTES", 10)
    with pytest.raises(lib.LibraryError, match="too big for a tracked shelf"):
        lib.Library(shelf(tmp_path)).add(run_dir / "OKLO_2026-09-15_繁體中文報告.pdf",
                                         ticker="OKLO", date_str="2026-09-15")


# ---------------------------------------------------------------------------------------------
# add
# ---------------------------------------------------------------------------------------------


def test_add_copies_bytes_and_derives_metadata_from_the_pdf(lib, tmp_path, run_dir):
    source = run_dir / "OKLO_2026-09-15_繁體中文報告.pdf"
    folder = shelf(tmp_path)
    entry, status = lib.Library(folder).add(source, ticker="OKLO", date_str="2026-09-15")
    assert status == "added"
    assert (folder / entry.file).read_bytes() == source.read_bytes()
    assert entry.pages == 1 and entry.sha256 and entry.bytes == source.stat().st_size
    assert entry.rating_label == "減碼"  # from the "rating: …" keyword the renderer writes
    assert entry.title == "交易分析報告：OKLO"


def test_add_records_markdown_and_the_reader_to_link(lib, tmp_path, run_dir):
    folder = shelf(tmp_path)
    entry, _ = lib.Library(folder).add(
        run_dir / "OKLO_2026-09-15_繁體中文報告.pdf", ticker="OKLO", date_str="2026-09-15",
        markdown=run_dir / "complete_report_zh.md",
        reader=run_dir / "OKLO_2026-09-15_繁體中文報告.html",
    )
    assert (folder / f"{entry.slug}.md").read_text(encoding="utf-8").startswith("# 交易分析報告")
    assert entry.reader.endswith("繁體中文報告.html")
    index = lib.Library(folder).render_index([entry])
    # relative to the shelf, so it still resolves under the sandbox's proxied preview path
    assert '../reports/OKLO_2026-09-15/' not in index
    assert '../OKLO_2026-09-15/' in index and '2026-09-15_OKLO_traditional_chinese.md' in index


def test_rating_falls_back_to_the_markdown_when_the_pdf_says_nothing(lib, tmp_path, run_dir):
    bare = tmp_path / "bare.pdf"
    make_pdf(bare)  # no title, no "rating:" keyword
    entry, _ = lib.Library(shelf(tmp_path)).add(
        bare, ticker="OKLO", date_str="2026-09-15", markdown=run_dir / "complete_report_zh.md"
    )
    assert entry.rating == "Underweight"  # **Rating**: Underweight in the PM section
    assert entry.title == "" and entry.pages == 1


def test_ticker_case_and_padding_are_normalised(lib, tmp_path, run_dir):
    """One shelf entry per report, not one per way of typing the ticker."""
    entry, _ = lib.Library(shelf(tmp_path)).add(
        run_dir / "OKLO_2026-09-15_繁體中文報告.pdf", ticker=" oklo ", date_str="2026-09-15"
    )
    assert (entry.ticker, entry.slug) == ("OKLO", "2026-09-15_OKLO_traditional_chinese")


def test_missing_source_is_an_error(lib, tmp_path):
    with pytest.raises(lib.LibraryError, match="no such PDF"):
        lib.Library(shelf(tmp_path)).add(tmp_path / "ghost.pdf", ticker="OKLO", date_str="2026-09-15")


# ---------------------------------------------------------------------------------------------
# idempotency, pruning, adoption
# ---------------------------------------------------------------------------------------------


def test_refiling_identical_bytes_is_unchanged_and_force_updates(lib, tmp_path, run_dir):
    folder = shelf(tmp_path)
    source = run_dir / "OKLO_2026-09-15_繁體中文報告.pdf"
    library = lib.Library(folder)
    entry, _ = library.add(source, ticker="OKLO", date_str="2026-09-15")

    again, status = lib.Library(folder).add(source, ticker="OKLO", date_str="2026-09-15")
    assert status == "unchanged" and again.pages == entry.pages

    make_pdf(source, title="重製後", rating="Hold", text="v2")  # re-rendered report
    updated, status = lib.Library(folder).add(source, ticker="OKLO", date_str="2026-09-15",
                                              force=True)
    assert status == "updated" and updated.rating == "Hold" and updated.title == "重製後"
    # one slug, one card — no duplicates on the shelf
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["entries"]) == 1


def test_build_prunes_deleted_files_and_adopts_dropped_ones(lib, tmp_path, run_dir):
    folder = shelf(tmp_path)
    library = lib.Library(folder)
    library.add(run_dir / "OKLO_2026-09-15_繁體中文報告.pdf", ticker="OKLO",
                date_str="2026-09-15", markdown=run_dir / "complete_report_zh.md")
    (folder / "2026-09-15_RKLB_traditional_chinese.pdf").write_bytes(
        make_pdf(tmp_path / "r.pdf", title="Trading Analysis: RKLB", rating="Hold").read_bytes())

    stats = library.build()
    assert stats["entries"] == 2 and stats["adopted"] == 1
    html = (folder / "index.html").read_text(encoding="utf-8")
    assert "Trading Analysis: RKLB" in html and "manually dropped in" in html

    (folder / "2026-09-15_RKLB_traditional_chinese.pdf").unlink()
    assert library.build()["entries"] == 1
    assert "RKLB" not in (folder / "index.html").read_text(encoding="utf-8")
    assert (folder / "README.md").read_text(encoding="utf-8").count(".pdf)") == 1


def test_dry_run_writes_nothing(lib, tmp_path, run_dir):
    folder = shelf(tmp_path)
    entry, status = lib.Library(folder, dry_run=True).add(
        run_dir / "OKLO_2026-09-15_繁體中文報告.pdf", ticker="OKLO", date_str="2026-09-15")
    assert status == "added" and entry.slug
    assert not folder.exists()


# ---------------------------------------------------------------------------------------------
# sync + CLI
# ---------------------------------------------------------------------------------------------


def test_sync_finds_only_run_dirs_with_a_zh_pdf(lib, tmp_path, run_dir):
    reports = run_dir.parent
    (reports / "notes").mkdir()
    (reports / "2026-09-15").mkdir()
    folder = shelf(tmp_path)
    done = lib.Library(folder).sync(reports)
    assert [(e.ticker, status) for e, status in done] == [("OKLO", "added")]
    assert done[0][0].name == "Oklo Inc. (NYSE: OKLO)"  # from pack.json's instrument_context
    assert not (folder / "notes").exists() and list(folder.glob("*.pdf"))

    # A run without a typeset 繁中 PDF is left alone rather than half-filed, and a run that is
    # already on the shelf is reported unchanged instead of duplicated.
    (reports / "SMR_2026-09-15").mkdir()
    assert [status for _, status in lib.Library(folder).sync(reports)] == ["unchanged"]
    assert len(list(folder.glob("*.pdf"))) == 1


def test_cli_end_to_end_and_exit_codes(lib, tmp_path, run_dir, capsys):
    pdf = run_dir / "OKLO_2026-09-15_繁體中文報告.pdf"
    folder = shelf(tmp_path)
    code = lib.main(["add", str(pdf), "--library", str(folder), "--ticker", "oklo",
                     "--date", "2026-09-15", "--markdown", str(run_dir / "complete_report_zh.md"),
                     "--no-index"])
    out = capsys.readouterr().out
    assert code == 0 and "added" in out and "減碼" in out
    capsys.readouterr()  # build below prints too; keep --json output parseable
    assert (folder / "2026-09-15_OKLO_traditional_chinese.pdf").is_file()

    assert lib.main(["build", "--library", str(folder)]) == 0
    capsys.readouterr()
    assert lib.main(["list", "--library", str(folder), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["ticker"] == "OKLO" and payload[0]["lang"] == "zh"

    assert lib.main(["add", str(run_dir / "pack.json"), "--library", str(folder),
                     "--ticker", "OKLO", "--date", "2026-09-15"]) == 1
    assert lib.main(["sync", "--library", str(folder), "--reports", str(tmp_path / "none")]) == 1
