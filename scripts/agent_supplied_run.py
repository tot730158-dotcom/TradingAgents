#!/usr/bin/env python3
"""Agent-supplied (keyless) run: drive the TradingAgents decision path without any LLM provider.

The graph's *plumbing* — analyst reports, the two debate records, the structured
Research Manager / Trader / Portfolio Manager blocks, the extracted 5-tier signal, the
report tree on disk, the state log, and the pending memory-log entry — is all produced
here from a JSON "run pack". What is NOT produced here is the reasoning: every section's
text is supplied by an external thinker (a local model, another agent runtime, a human
analyst, or a prompt file you filled in by hand). That makes this the offline counterpart
to ``tradingagents.graph.trading_graph.TradingAgentsGraph.propagate`` for environments with
no ``OPENAI_API_KEY`` / ``ANTHROPIC_API_KEY`` / network egress to a data vendor.

Nothing is re-implemented: the pack's structured blocks are validated by the very same
pydantic models the agents use (:class:`ResearchPlan`, :class:`TraderProposal`,
:PortfolioDecision, :class:`SentimentReport`), rendered by the same ``render_*`` helpers,
signalled through :class:`SignalProcessor`, written by
:func:`tradingagents.reporting.write_report_tree` and logged through
:class:`TradingMemoryLog`. So an externally-produced run is byte-for-byte shaped like an
LLM-produced one, and the guard rails the LLM path relies on (absolute-price entry/stop
coercion, the REVIEW sentinel for unparseable ratings) are enforced here too.

Usage
-----
    # 1. Emit a skeleton pack (fills in section files as you write them)
    python scripts/agent_supplied_run.py --template /tmp/RKLB_pack.json --ticker RKLB \
        --date 2026-09-15

    # 2. Fill the pack in, then validate only (no writes)
    python scripts/agent_supplied_run.py /tmp/RKLB_pack.json --validate-only

    # 3. Produce the run: report tree + state log + memory-log pending entry
    python scripts/agent_supplied_run.py /tmp/RKLB_pack.json --write-reports \
        --memory-log --state-log --out /tmp/reports/RKLB_20260915

Pack schema
-----------
Every text slot accepts either a literal string or ``{"file": "path.md"}`` (resolved
relative to the pack file, which is the convenient form: keep each agent's prose in its
own markdown file next to the pack). Required slots are the four analyst reports and both
debate records plus the three structured blocks; anything else is optional.

    {
      "ticker": "RKLB", "trade_date": "2026-09-15", "asset_type": "stock",
      "instrument_context": "…identity line the graph injects into every prompt…",
      "reports":   {"market": …, "sentiment": …, "news": …, "fundamentals": …},
      "sentiment": {"overall_band": "Mixed", "overall_score": 4.6,
                    "confidence": "medium", "narrative": …},   # overrides reports.sentiment
      "debate":    {"bull": …, "bear": …, "history": …},
      "research_plan":  {"recommendation": …, "rationale": …, "strategic_actions": …},
      "trader_proposal": {"action": …, "reasoning": …, "entry_price": …,
                          "stop_loss": …, "position_sizing": …},
      "risk_debate":  {"aggressive": …, "conservative": …, "neutral": …, "history": …},
      "pm_decision":  {"rating": …, "executive_summary": …, "investment_thesis": …,
                       "price_target": …, "time_horizon": …}
    }

Every section's prose is plain markdown, so a localised run (set
``TRADINGAGENTS_OUTPUT_LANGUAGE=繁體中文``) can be typeset straight to PDF with ``--pdf``:
it renders the report tree's ``complete_report.md`` through ``scripts/report_to_pdf.py``,
which ships a CJK-capable face and needs no TeX, weasyprint or system fonts. The PDF is
written next to the report tree as ``complete_report.pdf``.

Exit status: 0 on success, 1 if validation fails or the extracted signal is ``REVIEW``
(that decision text carries no recognizable 5-tier rating and must not be treated as
tradeable), 2 on a usage/IO error (including ``--pdf`` without PyMuPDF installed).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tradingagents.agents.schemas import (  # noqa: E402
    PortfolioDecision,
    ResearchPlan,
    SentimentReport,
    TraderProposal,
    render_pm_decision,
    render_research_plan,
    render_sentiment_report,
    render_trader_proposal,
)
from tradingagents.agents.utils.rating import RATING_REVIEW, extract_rating  # noqa: E402
from tradingagents.dataflows.utils import safe_ticker_component  # noqa: E402
from tradingagents.graph.signal_processing import SignalProcessor  # noqa: E402
from tradingagents.reporting import write_report_tree  # noqa: E402

# Slots that must be populated for the run to be a *complete* agent pass rather than a
# partial replay. write_report_tree tolerates missing sections, so without this check a
# pack that forgot the fundamentals analyst would still silently "succeed". Checked
# against the resolved state, so structured blocks and file-backed prose are treated alike.
REQUIRED_SLOTS: tuple[tuple[str, Any], ...] = (
    ("reports.market", lambda s: s["market_report"]),
    ("reports.news", lambda s: s["news_report"]),
    ("reports.fundamentals", lambda s: s["fundamentals_report"]),
    ("debate.bull", lambda s: s["investment_debate_state"]["bull_history"]),
    ("debate.bear", lambda s: s["investment_debate_state"]["bear_history"]),
    ("research_plan", lambda s: s["investment_plan"]),
    ("trader_proposal", lambda s: s["trader_investment_plan"]),
    ("risk_debate.aggressive", lambda s: s["risk_debate_state"]["aggressive_history"]),
    ("risk_debate.conservative", lambda s: s["risk_debate_state"]["conservative_history"]),
    ("risk_debate.neutral", lambda s: s["risk_debate_state"]["neutral_history"]),
    ("pm_decision", lambda s: s["final_trade_decision"]),
)

# Guidance bands documented on SentimentReport.overall_score. The structured agent does
# not enforce them (a 0-10 range is all pydantic checks), so an out-of-band score that
# contradicts the label slips into reports; here it becomes a visible warning.
_SENTIMENT_BANDS = {
    "Bullish": (6.5, 10.0),
    "Mildly Bullish": (5.5, 6.4),
    "Neutral": (4.5, 5.5),
    "Mixed": (4.5, 5.5),
    "Mildly Bearish": (3.5, 4.4),
    "Bearish": (0.0, 3.4),
}


class PackError(ValueError):
    """Raised when the run pack cannot be turned into a valid agent pass."""


def _resolve(value: Any, base: Path) -> str:
    """Resolve a pack text slot: literal string, {"file": …}, or {"text": …}."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        if "text" in value:
            return str(value["text"]).strip()
        if "file" in value:
            path = Path(value["file"])
            if not path.is_absolute():
                path = base / path
            try:
                return path.read_text(encoding="utf-8").strip()
            except OSError as exc:
                raise PackError(f"could not read pack section file {path}: {exc}") from exc
    raise PackError(f"unsupported pack slot value {value!r}; expected a string or {{'file': …}}")


def _slot(pack: dict[str, Any], dotted: str, base: Path) -> str:
    node: Any = pack
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return ""
        node = node[part]
    return _resolve(node, base)


def _resolve_payload(payload: dict[str, Any], base: Path) -> dict[str, Any]:
    """Resolve ``{"file": ...}`` slots nested inside a structured block.

    Long prose belongs in markdown files next to the pack, but pydantic wants the string.
    """
    resolved = {}
    for key, value in payload.items():
        if isinstance(value, dict) and ("file" in value or "text" in value):
            resolved[key] = _resolve(value, base)
        else:
            resolved[key] = value
    return resolved


def _validate(model, payload: dict[str, Any], label: str, notes: list[str]):
    try:
        return model.model_validate(payload)
    except Exception as exc:  # pydantic ValidationError and friends
        raise PackError(f"{label} failed schema validation: {exc}") from exc


def _price_sanity(proposal: TraderProposal, notes: list[str], warnings: list[str]) -> None:
    """Deterministic checks on the trader's absolute price levels.

    entry_price/stop_loss arrive as free-form model text, so pydantic's coercion
    (:func:`tradingagents.agents.schemas._coerce_optional_float`) already turns "N/A" and
    "15%" into None. Two failures survive that: a percentage that was *dropped* (the
    proposal quietly loses its level) and a stop on the wrong side of entry.
    """
    for field in ("entry_price", "stop_loss"):
        if getattr(proposal, field) is None:
            continue
        value = float(getattr(proposal, field))
        if value <= 0:
            warnings.append(f"trader {field}={value} is not a positive price")
    if proposal.entry_price and proposal.stop_loss:
        entry, stop = float(proposal.entry_price), float(proposal.stop_loss)
        if proposal.action.value in {"Buy"} and stop >= entry:
            warnings.append(
                f"stop {stop} is at or above entry {entry} on a Buy — the position has no "
                "defined risk; put the stop below the entry (or below the cited support)"
            )
        if proposal.action.value == "Sell" and stop <= entry:
            warnings.append(
                f"stop {stop} is at or below entry {entry} on a Sell — a short stop belongs "
                "above the entry"
            )
        if proposal.action.value == "Hold" and proposal.entry_price:
            notes.append(
                "Hold carries an entry price; read it as the trigger level for a staged "
                "re-entry rather than an immediate fill"
            )


def build_state(pack: dict[str, Any], base: Path) -> tuple[dict[str, Any], list[str], list[str]]:
    """Turn a run pack into the graph's final state, exactly as the nodes would."""
    notes: list[str] = []
    warnings: list[str] = []

    ticker = str(pack.get("ticker") or "").strip()
    trade_date = str(pack.get("trade_date") or "").strip()
    if not ticker or not trade_date:
        raise PackError("pack must set 'ticker' and 'trade_date'")

    state: dict[str, Any] = {
        "company_of_interest": ticker,
        "trade_date": trade_date,
        "asset_type": str(pack.get("asset_type") or "stock"),
        "instrument_context": _slot(pack, "instrument_context", base),
        "sender": "Agent Supplied Run",
        "market_report": _slot(pack, "reports.market", base),
        "news_report": _slot(pack, "reports.news", base),
        "fundamentals_report": _slot(pack, "reports.fundamentals", base),
        "sentiment_report": _slot(pack, "reports.sentiment", base),
    }

    # Sentiment: the structured report wins over free text, mirroring the agent's
    # structured-then-freetext fallback, and is rendered with the shared renderer so the
    # saved markdown carries the same deterministic band/score header.
    sentiment_raw = pack.get("sentiment")
    if isinstance(sentiment_raw, dict) and (
        "narrative" in sentiment_raw or "overall_band" in sentiment_raw
    ):
        report = _validate(SentimentReport, _resolve_payload(sentiment_raw, base), "sentiment report", notes)
        state["sentiment_report"] = render_sentiment_report(report)
        band_range = _SENTIMENT_BANDS.get(report.overall_band.value)
        if band_range and not (band_range[0] <= report.overall_score <= band_range[1]):
            msg = (
                f"sentiment score {report.overall_score:.1f}/10 is outside the "
                f"{report.overall_band.value} band {band_range[0]}-{band_range[1]}"
            )
            warnings.append(msg)

    debate_history = _slot(pack, "debate.history", base)
    bull = _slot(pack, "debate.bull", base)
    bear = _slot(pack, "debate.bear", base)

    plan_payload = pack.get("research_plan")
    if isinstance(plan_payload, dict):
        plan = _validate(ResearchPlan, _resolve_payload(plan_payload, base), "research plan", notes)
        investment_plan = render_research_plan(plan)
    else:
        investment_plan = _resolve(plan_payload, base)
        if investment_plan:
            notes.append("research_plan supplied as prose; the Research Manager runs structured in the graph")

    trader_payload = pack.get("trader_proposal")
    if isinstance(trader_payload, dict):
        proposal = _validate(TraderProposal, _resolve_payload(trader_payload, base), "trader proposal", notes)
        trader_plan = render_trader_proposal(proposal)
        _price_sanity(proposal, notes, warnings)
    else:
        trader_plan = _resolve(trader_payload, base)
        proposal = None
        if trader_plan:
            notes.append("trader_proposal supplied as prose; entry/stop coercion checks were skipped")

    risk = {
        "aggressive_history": _slot(pack, "risk_debate.aggressive", base),
        "conservative_history": _slot(pack, "risk_debate.conservative", base),
        "neutral_history": _slot(pack, "risk_debate.neutral", base),
        "history": _slot(pack, "risk_debate.history", base),
    }
    # Rebuild the transcript the risk analysts would have accumulated, in the graph's
    # speaking order (Aggressive -> Conservative -> Neutral), when the pack omits it.
    if not risk["history"]:
        risk["history"] = "\n\n".join(
            f"{name}:\n{text}"
            for name, text in (
                ("Aggressive Analyst", risk["aggressive_history"]),
                ("Conservative Analyst", risk["conservative_history"]),
                ("Neutral Analyst", risk["neutral_history"]),
            )
            if text
        )

    pm_payload = pack.get("pm_decision")
    if isinstance(pm_payload, dict):
        decision = _validate(PortfolioDecision, _resolve_payload(pm_payload, base), "portfolio manager decision", notes)
        final_trade_decision = render_pm_decision(decision)
    else:
        final_trade_decision = _resolve(pm_payload, base)
        if final_trade_decision:
            notes.append("pm_decision supplied as prose; only the rating-extraction checks ran")

    state.update(
        {
            "investment_debate_state": {
                "bull_history": bull,
                "bear_history": bear,
                "history": debate_history or f"Bull Researcher:\n{bull}\n\nBear Researcher:\n{bear}",
                "current_response": investment_plan,
                "judge_decision": investment_plan,
                "count": 2 * int(bool(bull)) + 2 * int(bool(bear)),
            },
            "investment_plan": investment_plan,
            "trader_investment_plan": trader_plan,
            "risk_debate_state": {
                **risk,
                "latest_speaker": "Neutral Analyst",
                "current_response": final_trade_decision,
                "judge_decision": final_trade_decision,
                "count": 3,
            },
            "final_trade_decision": final_trade_decision,
            "past_context": _slot(pack, "past_context", base),
        }
    )

    missing = [label for label, read in REQUIRED_SLOTS if not read(state).strip()]
    if missing:
        warnings.append("empty required slot(s): " + ", ".join(missing))
    if proposal is not None and isinstance(trader_payload, dict):
        for field in ("entry_price", "stop_loss"):
            raw = _resolve_payload(trader_payload, base).get(field)
            if isinstance(raw, str) and getattr(proposal, field) is None:
                warnings.append(
                    f"trader {field}={raw!r} was dropped by the optional-float coercion; "
                    "state an absolute price or omit the field"
                )
    return state, notes, warnings


def summarize(state: dict[str, Any], report_path: Path | None) -> str:
    signal = SignalProcessor().process_signal(state["final_trade_decision"])
    lines = [
        f"ticker           : {state['company_of_interest']}",
        f"trade date       : {state['trade_date']}",
        f"asset type       : {state['asset_type']}",
        f"extracted signal : {signal}",
        f"research manager : {extract_rating(state['investment_plan']) or RATING_REVIEW}",
        "",
        "sections (chars written):",
    ]
    for label, key in (
        ("market", "market_report"),
        ("sentiment", "sentiment_report"),
        ("news", "news_report"),
        ("fundamentals", "fundamentals_report"),
    ):
        lines.append(f"  analyst/{label:<12}: {len(state.get(key) or '')}")
    lines.append(f"  research/bull         : {len(state['investment_debate_state']['bull_history'])}")
    lines.append(f"  research/bear         : {len(state['investment_debate_state']['bear_history'])}")
    lines.append(f"  trading/trader        : {len(state['trader_investment_plan'])}")
    lines.append(f"  portfolio/decision    : {len(state['final_trade_decision'])}")
    if report_path is not None:
        lines += ["", f"complete report      : {report_path}"]
    if signal == RATING_REVIEW:
        lines.append(
            "  ! REVIEW: the Portfolio Manager text has no recognizable Buy/Overweight/Hold/"
            "Underweight/Sell rating, so it is not tradeable — fix pm_decision and re-run"
        )
    return "\n".join(lines)


def _write_template(ticker: str, trade_date: str) -> dict[str, Any]:
    slot = {"file": "SECTION.md"}
    return {
        "ticker": ticker,
        "trade_date": trade_date,
        "asset_type": "stock",
        "instrument_context": "",
        "reports": {k: dict(slot) for k in ("market", "sentiment", "news", "fundamentals")},
        "sentiment": {
            "overall_band": "Mixed",
            "overall_score": 5.0,
            "confidence": "medium",
            "narrative": dict(slot),
        },
        "debate": {"bull": dict(slot), "bear": dict(slot), "history": ""},
        "research_plan": {
            "recommendation": "Hold",
            "rationale": dict(slot),
            "strategic_actions": dict(slot),
        },
        "trader_proposal": {
            "action": "Hold",
            "reasoning": dict(slot),
            "entry_price": None,
            "stop_loss": None,
            "position_sizing": "",
        },
        "risk_debate": {
            "aggressive": dict(slot),
            "conservative": dict(slot),
            "neutral": dict(slot),
            "history": "",
        },
        "pm_decision": {
            "rating": "Hold",
            "executive_summary": dict(slot),
            "investment_thesis": dict(slot),
            "price_target": None,
            "time_horizon": "",
        },
    }


def load_pdf_renderer():
    """Import ``scripts/report_to_pdf.py`` by path (``scripts`` is not a package)."""
    if importlib.util.find_spec("pymupdf") is None:
        raise PackError(
            "PDF output needs PyMuPDF: pip install \"tradingagents[pdf]\" "
            "(or: pip install pymupdf)"
        )
    path = Path(__file__).with_name("report_to_pdf.py")
    spec = importlib.util.spec_from_file_location("report_to_pdf", path)
    if spec is None or spec.loader is None:  # pragma: no cover - broken install
        raise PackError(f"cannot load the PDF renderer at {path}")
    module = importlib.util.module_from_spec(spec)
    # Register before exec: the renderer's @dataclass classes resolve annotations through
    # sys.modules[cls.__module__], which a bare spec-load leaves unset.
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    return module


def render_pdf_report(report: Path, *, lang: str, page_size: str) -> dict:
    """Typeset one report markdown file into a PDF next to it."""
    module = load_pdf_renderer()
    return module.render_pdf(report, report.with_name(f"{report.stem}.pdf"),
                             lang=lang, page_size=page_size)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the TradingAgents decision path from an externally supplied pack (no API key).",
    )
    parser.add_argument("pack", nargs="?", type=Path, help="path to the run pack JSON")
    parser.add_argument("--template", type=Path, metavar="OUT.json", help="write a skeleton pack and exit")
    parser.add_argument("--ticker", default="TICKER", help="template ticker")
    parser.add_argument("--date", default=datetime.now().strftime("%Y-%m-%d"), help="template trade date")
    parser.add_argument("--out", type=Path, help="report-tree directory (default: results_dir/reports/<TICKER>_<stamp>)")
    parser.add_argument("--validate-only", action="store_true", help="check the pack and print the signal; write nothing")
    parser.add_argument("--write-reports", action="store_true", help="write the report tree via write_report_tree")
    parser.add_argument("--memory-log", action="store_true", help="append the pending entry to the decision log")
    parser.add_argument("--state-log", action="store_true", help="write full_states_log_<date>.json like the graph does")
    parser.add_argument("--pdf", action="store_true",
                        help="also typeset complete_report.md as a paginated PDF (needs pymupdf)")
    parser.add_argument("--pdf-lang", default="auto", choices=("auto", "en", "zh", "ja", "ko"),
                        help="language of the PDF chrome (cover caption, rating label, footer); "
                             "default: detected from each report's script")
    parser.add_argument("--pdf-page-size", default="a4", choices=("a4", "letter", "a5"),
                        help="PDF page size")
    parser.add_argument("--strict", action="store_true", help="treat warnings as errors")
    args = parser.parse_args(argv)

    if args.template:
        args.template.parent.mkdir(parents=True, exist_ok=True)
        args.template.write_text(
            json.dumps(_write_template(args.ticker, args.date), indent=2) + "\n", encoding="utf-8"
        )
        print(f"wrote skeleton pack: {args.template}")
        print("fill every SECTION.md slot (or replace them with inline strings), then run with --validate-only")
        return 0

    if args.pack is None:
        parser.error("a pack path is required (or use --template)")
    try:
        pack = json.loads(args.pack.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"error: cannot read pack {args.pack}: {exc}", file=sys.stderr)
        return 2

    from tradingagents.default_config import DEFAULT_CONFIG

    try:
        state, notes, warnings = build_state(pack, args.pack.resolve().parent)
    except PackError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    for line in notes:
        print(f"note: {line}")
    for line in warnings:
        print(f"warning: {line}", file=sys.stderr)

    if args.validate_only:
        report_path = None
    else:
        save_path = args.out
        if save_path is None:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = (
                Path(DEFAULT_CONFIG["results_dir"])
                / "reports"
                / f"{safe_ticker_component(state['company_of_interest'])}_{stamp}"
            )
        if args.write_reports:
            report_path = write_report_tree(state, state["company_of_interest"], save_path)
        else:
            report_path = None

    if args.pdf:
        if report_path is None:
            print("error: --pdf needs --write-reports (it typesets the written complete_report.md)",
                  file=sys.stderr)
            return 2
        try:
            info = render_pdf_report(Path(report_path), lang=args.pdf_lang, page_size=args.pdf_page_size)
        except PackError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(
            f"pdf: {info['path']}  ({info['pages']} pages, lang {info['lang']}, "
            f"rating {info['rating']}, {info['bytes'] / 1024:.0f} KB)"
        )

    if args.state_log:
        directory = (
            Path(DEFAULT_CONFIG["results_dir"])
            / safe_ticker_component(state["company_of_interest"])
            / "TradingAgentsStrategy_logs"
        )
        directory.mkdir(parents=True, exist_ok=True)
        payload = {k: v for k, v in state.items() if k != "past_context"}
        (directory / f"full_states_log_{state['trade_date']}.json").write_text(
            json.dumps(payload, indent=4), encoding="utf-8"
        )

    if args.memory_log:
        from tradingagents.agents.utils.memory import TradingMemoryLog

        log = TradingMemoryLog(DEFAULT_CONFIG)
        log.store_decision(
            state["company_of_interest"],
            state["trade_date"],
            state["final_trade_decision"],
        )
        print(f"memory log: pending entry stored for {state['trade_date']} {state['company_of_interest']}")

    print()
    print(summarize(state, report_path))

    if args.strict and warnings:
        print("\nstrict mode: warnings present", file=sys.stderr)
        return 1
    if state["final_trade_decision"] and extract_rating(state["final_trade_decision"]) is None:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
