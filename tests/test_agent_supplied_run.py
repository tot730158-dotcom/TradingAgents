"""Keyless (agent-supplied) run path: an externally produced pack must land in exactly the
shapes the LLM path produces — validated by the repo's own pydantic schemas, rendered by the
shared renderers, signalled through the REVIEW-aware extractor, and written by the shared
report writer. This is the offline counterpart to ``propagate`` for environments with no API
key or no market-data egress."""

import importlib.util
import json
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "agent_supplied_run.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("agent_supplied_run", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = pytest.fixture(scope="module")(_load_module)


def _pack(**overrides):
    pack = {
        "ticker": "RKLB",
        "trade_date": "2026-09-15",
        "reports": {
            "market": "TREND DOWN",
            "news": "FOMC TOMORROW",
            "fundamentals": "RECORD REVENUE, LOWER GM GUIDE",
        },
        "debate": {"bull": "BULL", "bear": "BEAR"},
        "research_plan": {
            "recommendation": "Hold",
            "rationale": "Business improving, financing and calendar are not.",
            "strategic_actions": "Stage in halves above 67.50; protect 58.00.",
        },
        "trader_proposal": {
            "action": "Hold",
            "reasoning": "Downtrend into a policy meeting.",
            "entry_price": 68.50,
            "stop_loss": 57.80,
            "position_sizing": "3% of NAV in halves",
        },
        "risk_debate": {"aggressive": "BUY", "conservative": "REDUCE", "neutral": "HOLD"},
        "pm_decision": {
            "rating": "Hold",
            "executive_summary": "Add nothing before the FOMC or the IRDM vote.",
            "investment_thesis": "46x EV/Sales on guided margin contraction.",
            "price_target": 80.00,
            "time_horizon": "3-6 months",
        },
    }
    pack.update(overrides)
    return pack


@pytest.mark.unit
def test_structured_blocks_render_like_the_agent_path(tmp_path, runner):
    state, notes, warnings = runner.build_state(_pack(), tmp_path)

    # The repo's renderers, not the pack's raw dicts, are what lands in state.
    assert state["final_trade_decision"].startswith("**Rating**: Hold")
    assert "**Price Target**: 80.0" in state["final_trade_decision"]
    assert state["trader_investment_plan"].endswith("FINAL TRANSACTION PROPOSAL: **HOLD**")
    assert state["investment_plan"].startswith("**Recommendation**: Hold")
    assert state["investment_debate_state"]["judge_decision"] == state["investment_plan"]
    # A Hold that still quotes an entry level is the staged-re-entry shape, so the
    # runner annotates rather than rejects it; nothing in a valid pack should warn.
    assert any("trigger level" in n for n in notes)
    assert warnings == []


@pytest.mark.unit
def test_risk_transcript_is_synthesised_in_graph_speaking_order(tmp_path, runner):
    state, _, _ = runner.build_state(_pack(), tmp_path)
    history = state["risk_debate_state"]["history"]
    assert history.index("Aggressive Analyst") < history.index("Conservative Analyst") < history.index(
        "Neutral Analyst"
    )


@pytest.mark.unit
def test_file_backed_slots_resolve_against_the_pack_directory(tmp_path, runner):
    (tmp_path / "1_analysts").mkdir()
    (tmp_path / "1_analysts" / "market.md").write_text("# Market\n\nBELOW ALL MVS\n", encoding="utf-8")
    pack = _pack()
    pack["reports"]["market"] = {"file": "1_analysts/market.md"}

    state, _, warnings = runner.build_state(pack, tmp_path)

    assert state["market_report"] == "# Market\n\nBELOW ALL MVS"
    assert warnings == []


@pytest.mark.unit
def test_nested_file_slots_resolve_inside_structured_blocks(tmp_path, runner):
    (tmp_path / "thesis.md").write_text("  46x EV/Sales into guided margin contraction.  \n", encoding="utf-8")
    pack = _pack()
    pack["pm_decision"]["investment_thesis"] = {"file": "thesis.md"}

    state, _, warnings = runner.build_state(pack, tmp_path)

    assert "**Investment Thesis**: 46x EV/Sales into guided margin contraction." in state[
        "final_trade_decision"
    ]
    assert warnings == []


@pytest.mark.unit
def test_missing_required_slot_is_reported(tmp_path, runner):
    pack = _pack()
    del pack["reports"]["fundamentals"]

    _, _, warnings = runner.build_state(pack, tmp_path)

    assert any("reports.fundamentals" in w for w in warnings)


@pytest.mark.unit
def test_formatted_price_is_parsed_and_a_percentage_level_is_dropped(tmp_path, runner):
    pack = _pack()
    pack["trader_proposal"]["entry_price"] = "$68.50"
    pack["trader_proposal"]["stop_loss"] = "12%"

    state, _, warnings = runner.build_state(pack, tmp_path)

    assert "**Entry Price**: 68.5" in state["trader_investment_plan"]
    # A percentage cannot be salvaged into an absolute level (#1288), so the field is
    # nulled rather than misread — but the pack must not lose it silently.
    assert "**Stop Loss**" not in state["trader_investment_plan"]
    assert any("stop_loss='12%' was dropped" in w for w in warnings)


@pytest.mark.unit
def test_buy_stop_above_entry_is_flagged(tmp_path, runner):
    pack = _pack()
    pack["trader_proposal"]["action"] = "Buy"
    pack["trader_proposal"]["entry_price"] = 62.5
    pack["trader_proposal"]["stop_loss"] = 66.0

    _, _, warnings = runner.build_state(pack, tmp_path)

    assert any("no defined risk" in w for w in warnings)


@pytest.mark.unit
def test_sentiment_report_header_is_rendered_and_band_score_mismatch_warns(tmp_path, runner):
    pack = _pack()
    pack["sentiment"] = {
        "overall_band": "Mixed",
        "overall_score": 4.6,
        "confidence": "medium",
        "narrative": "Sources point in different directions.",
    }
    state, _, warnings = runner.build_state(pack, tmp_path)
    assert state["sentiment_report"].startswith("**Overall Sentiment:** **Mixed** (Score: 4.6/10)")
    assert warnings == []

    pack["sentiment"]["overall_score"] = 2.1  # Bearish-band score on a Mixed label
    state, _, warnings = runner.build_state(pack, tmp_path)
    assert any("outside the Mixed band" in w for w in warnings)


@pytest.mark.unit
def test_unparseable_rating_exits_nonzero_as_review(tmp_path, runner, capsys):
    pack = _pack()
    pack["pm_decision"] = "We are not comfortable committing to a direction this week."
    pack_path = tmp_path / "pack.json"
    pack_path.write_text(json.dumps(pack), encoding="utf-8")

    code = runner.main([str(pack_path), "--validate-only"])

    assert code == 1
    out = capsys.readouterr().out
    assert "extracted signal : REVIEW" in out


@pytest.mark.unit
def test_bad_enum_fails_validation_with_a_named_error(tmp_path, runner):
    pack = _pack()
    pack["pm_decision"]["rating"] = "Strong Buy Please"
    with pytest.raises(runner.PackError, match="portfolio manager decision"):
        runner.build_state(pack, tmp_path)


@pytest.mark.unit
def test_full_run_writes_report_tree_and_pending_memory_entry(tmp_path, runner, capsys, monkeypatch):
    from tradingagents.default_config import DEFAULT_CONFIG

    monkeypatch.setitem(DEFAULT_CONFIG, "results_dir", str(tmp_path / "results"))
    pack_dir = tmp_path / "pack"
    pack_dir.mkdir()
    (pack_dir / "pack.json").write_text(json.dumps(_pack()), encoding="utf-8")
    out = tmp_path / "reports" / "RKLB_20260915"
    memory = tmp_path / "memory" / "trading_memory.md"
    monkeypatch.setitem(DEFAULT_CONFIG, "memory_log_path", str(memory))

    code = runner.main(
        [
            str(pack_dir / "pack.json"),
            "--write-reports",
            "--state-log",
            "--memory-log",
            "--out",
            str(out),
        ]
    )

    assert code == 0
    complete = out / "complete_report.md"
    assert complete.exists()
    text = complete.read_text()
    assert "## I. Analyst Team Reports" in text
    assert "## V. Portfolio Manager Decision" in text
    assert (out / "1_analysts" / "market.md").read_text() == "TREND DOWN"
    assert "**Entry Price**: 68.5" in text  # trader block reaches the report rendered
    state_log = (
        Path(DEFAULT_CONFIG["results_dir"])
        / "RKLB"
        / "TradingAgentsStrategy_logs"
        / "full_states_log_2026-09-15.json"
    )
    assert json.loads(state_log.read_text())["company_of_interest"] == "RKLB"
    # The memory log stores the same pending entry shape the graph writes.
    entry = memory.read_text()
    assert entry.startswith("[2026-09-15 | RKLB | Hold | pending]")
    assert "**Rating**: Hold" in entry
    assert "extracted signal : Hold" in capsys.readouterr().out


@pytest.mark.unit
def test_pdf_flag_typesets_the_written_report_tree(tmp_path, runner, capsys):
    pymupdf = pytest.importorskip("pymupdf")
    pack_dir = tmp_path / "pack"
    pack_dir.mkdir()
    (pack_dir / "pack.json").write_text(json.dumps(_pack()), encoding="utf-8")
    out = tmp_path / "reports" / "RKLB_20260915"

    code = runner.main(
        [str(pack_dir / "pack.json"), "--write-reports", "--pdf", "--pdf-lang", "zh", "--out", str(out)]
    )

    assert code == 0
    pdf = out / "complete_report.pdf"
    assert pdf.exists() and pdf.stat().st_size > 1000
    assert "pdf: " in capsys.readouterr().out
    with pymupdf.open(pdf) as doc:
        assert doc.page_count >= 2
        # Chinese chrome is a renderer-side choice; the report prose stays as written.
        assert "最終決策" in doc[0].get_text()


@pytest.mark.unit
def test_pdf_without_write_reports_is_a_usage_error(tmp_path, runner, capsys):
    pytest.importorskip("pymupdf")
    pack_dir = tmp_path / "pack"
    pack_dir.mkdir()
    (pack_dir / "pack.json").write_text(json.dumps(_pack()), encoding="utf-8")

    assert runner.main([str(pack_dir / "pack.json"), "--validate-only", "--pdf"]) == 2
    assert "--pdf needs --write-reports" in capsys.readouterr().err
