import textwrap

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from tabulate import tabulate

from .schema import FundamentalsSchema, MoatSchema, RiskSchema
from .state import State
from .subagents import fundamentals_subgraph, moat_subgraph, risk_subgraph



def generate_report(ticker: str, f: FundamentalsSchema, m: MoatSchema, r: RiskSchema) -> str:
    """Transform structured outputs into a markdown report without using LLM."""
    total = f.total_score + m.score + r.score

    # ── Fundamentals section ──────────────────────────────────────────────────
    eps_hist  = ", ".join(f"{v:.2f}" for v in f.eps_history) or "N/A"
    fcf_hist  = ", ".join(f"{v:.2f}" for v in f.free_cashflow_history) or "N/A"
    div_hist  = ", ".join(f"{v:.2f}" for v in f.dividend_history) or "N/A"
    nm_hist   = ", ".join(f"{v:.2f}" for v in f.net_margin_history) or "N/A"
    roic_hist = ", ".join(f"{v:.2f}" for v in f.roic_history) or "N/A"

    # ── Moat section ──────────────────────────────────────────────────────────
    moat_details = "\n".join(
        f"  - **{k}**: {v}" for k, v in m.explain.items()
    ) or "  - N/A"
    moat_sources = "\n".join(f"  - {s}" for s in m.source) or "  - N/A"

    # ── Risk section ──────────────────────────────────────────────────────────
    risk_details = "\n".join(
        f"  - **{k}**: {v}" for k, v in r.explain.items()
    ) or "  - N/A"
    risk_sources = "\n".join(f"  - {s}" for s in r.source) or "  - N/A"

    return f"""# Stock Research Report: {ticker.upper()}

---

## 📊 Overall Score: {total:.1f} / 10
| Category | Score |
|---|---|
| Fundamentals | {f.total_score:.1f} / 6 |
| Moat | {m.score} / 4 |
| Risk | {r.score} / 0 |
| **Total** | **{total:.1f} / 10** |

---

## 1. Fundamentals Analysis ({f.total_score:.1f} / 6)

| Metric | Score | Data (past 10 years) |
|---|---|---|
| EPS (USD) | {f.eps_score} / 1 | {eps_hist} |
| Free Cash Flow (M USD) | {f.free_cashflow_score} / 1 | {fcf_hist} |
| Dividend Per Share (USD) | {f.dividend_score} / 1 | {div_hist} |
| Net Margin (%) | {f.net_margin_score} / 1 | {nm_hist} |
| LT Debt / Equity (latest) | {f.debt_equity_score} / 1 | {f.debt_equity_history:.2f} |
| ROIC (%) | {f.roic_score} / 1 | {roic_hist} |

**Explanation:**
{f.explaination}

---

## 2. Moat Analysis ({m.score} / 4)

**Moat Types:** {', '.join(m.keywords) if m.keywords else 'None'}

{moat_details}

**Sources:**
{moat_sources}

---

## 3. Risk Analysis ({r.score} / 0)

**Risk Types:** {', '.join(r.keywords) if r.keywords else 'None'}

{risk_details}

**Sources:**
{risk_sources}
"""


def generate_cli_report(ticker: str, f: FundamentalsSchema, m: MoatSchema, r: RiskSchema) -> str:
    """Transform structured outputs into a CLI-friendly report using tabulate."""
    total = f.total_score + m.score + r.score
    width = 72

    # ── Header ───────────────────────────────────────────────────────────────
    lines: list[str] = []
    lines.append("=" * width)
    lines.append(f"  STOCK RESEARCH REPORT: {ticker.upper()}")
    lines.append("=" * width)

    # ── Overall score ─────────────────────────────────────────────────────────
    lines.append("")
    lines.append(tabulate(
        [
            ["Fundamentals", f"{f.total_score:.1f} / 6"],
            ["Moat",         f"{m.score} / 4"],
            ["Risk",         f"{r.score} / 0"],
            ["TOTAL",        f"{total:.1f} / 10"],
        ],
        headers=["Category", "Score"],
        tablefmt="rounded_outline",
        colalign=("left", "right"),
    ))

    # ── Fundamentals ──────────────────────────────────────────────────────────
    lines.append("")
    lines.append(f"  1. FUNDAMENTALS  ({f.total_score:.1f} / 6)")
    lines.append("-" * width)

    def _hist(values: list[float]) -> str:
        return ", ".join(f"{v:.2f}" for v in values) or "N/A"

    lines.append(tabulate(
        [
            ["EPS (USD)",            f"{f.eps_score} / 1",          _hist(f.eps_history)],
            ["Free Cash Flow (M USD)", f"{f.free_cashflow_score} / 1", _hist(f.free_cashflow_history)],
            ["Dividend/Share (USD)", f"{f.dividend_score} / 1",      _hist(f.dividend_history)],
            ["Net Margin (%)",       f"{f.net_margin_score} / 1",    _hist(f.net_margin_history)],
            ["LT Debt / Equity",     f"{f.debt_equity_score} / 1",  f"{f.debt_equity_history:.2f}"],
            ["ROIC (%)",             f"{f.roic_score} / 1",          _hist(f.roic_history)],
        ],
        headers=["Metric", "Score", "Data (past 10 years)"],
        tablefmt="rounded_outline",
    ))
    lines.append("")
    lines.append("  Explanation:")
    for para in f.explaination.splitlines():
        lines.append(textwrap.fill(para, width=width, initial_indent="  ", subsequent_indent="  "))

    # ── Moat ──────────────────────────────────────────────────────────────────
    lines.append("")
    lines.append(f"  2. MOAT  ({m.score} / 4)")
    lines.append("-" * width)
    lines.append(f"  Types: {', '.join(m.keywords) if m.keywords else 'None'}")
    lines.append("")
    if m.explain:
        moat_rows = [
            [k, textwrap.fill(v, width=46)]
            for k, v in m.explain.items()
        ]
        lines.append(tabulate(moat_rows, headers=["Moat Type", "Detail"], tablefmt="rounded_outline"))
    lines.append("")
    lines.append("  Sources:")
    for s in m.source:
        lines.append(f"  • {s}")

    # ── Risk ──────────────────────────────────────────────────────────────────
    lines.append("")
    lines.append(f"  3. RISK  ({r.score} / 0)")
    lines.append("-" * width)
    lines.append(f"  Types: {', '.join(r.keywords) if r.keywords else 'None'}")
    lines.append("")
    if r.explain:
        risk_rows = [
            [k, textwrap.fill(v, width=46)]
            for k, v in r.explain.items()
        ]
        lines.append(tabulate(risk_rows, headers=["Risk Type", "Detail"], tablefmt="rounded_outline"))
    lines.append("")
    lines.append("  Sources:")
    for s in r.source:
        lines.append(f"  • {s}")

    lines.append("")
    lines.append("=" * width)
    return "\n".join(lines)


def run_fundamentals_node(state: State):
    """Run fundamentals subgraph and store result."""
    result = fundamentals_subgraph.invoke({"ticker": state["ticker"]})
    return {"fundamentals_result": result["fundamentals_result"]}


def route_after_fundamentals(state: State):
    """Route based on fundamentals result; fan out in parallel if score >= 4."""
    f = state["fundamentals_result"]
    if not f.stock_exists:
        return "end_not_found"
    if f.total_score < 3:
        return "end_low_score"
    return [
        Send("run_moat_research", {"ticker": state["ticker"]}),
        Send("run_risk_research", {"ticker": state["ticker"]}),
    ]


def run_moat_research_node(state: dict):
    result = moat_subgraph.invoke({"ticker": state["ticker"]})
    return {"moat_result": result["moat_result"]}


def run_risk_research_node(state: dict):
    result = risk_subgraph.invoke({"ticker": state["ticker"]})
    return {"risk_result": result["risk_result"]}


def aggregate_node(state: State):
    """Aggregate all results into a markdown and CLI report."""
    f, m, r = state["fundamentals_result"], state["moat_result"], state["risk_result"]
    md_report = generate_report(state["ticker"], f, m, r)
    cli_report = generate_cli_report(state["ticker"], f, m, r)
    return {"md_report": md_report, "cli_report": cli_report}


def _fundamentals_md(ticker: str, f: FundamentalsSchema) -> str:
    eps_hist  = ", ".join(f"{v:.2f}" for v in f.eps_history) or "N/A"
    fcf_hist  = ", ".join(f"{v:.2f}" for v in f.free_cashflow_history) or "N/A"
    div_hist  = ", ".join(f"{v:.2f}" for v in f.dividend_history) or "N/A"
    nm_hist   = ", ".join(f"{v:.2f}" for v in f.net_margin_history) or "N/A"
    roic_hist = ", ".join(f"{v:.2f}" for v in f.roic_history) or "N/A"
    return f"""## 1. Fundamentals Analysis ({f.total_score:.1f} / 6)

| Metric | Score | Data (past 10 years) |
|---|---|---|
| EPS (USD) | {f.eps_score} / 1 | {eps_hist} |
| Free Cash Flow (M USD) | {f.free_cashflow_score} / 1 | {fcf_hist} |
| Dividend Per Share (USD) | {f.dividend_score} / 1 | {div_hist} |
| Net Margin (%) | {f.net_margin_score} / 1 | {nm_hist} |
| LT Debt / Equity (latest) | {f.debt_equity_score} / 1 | {f.debt_equity_history:.2f} |
| ROIC (%) | {f.roic_score} / 1 | {roic_hist} |

**Explanation:**
{f.explaination}"""


def end_low_score_node(state: State):
    f = state["fundamentals_result"]
    ticker = state["ticker"].upper()
    msg = f"Fundamentals score is {f.total_score:.1f}/6 (below 4). The company does not meet the minimum financial criteria."
    return {
        "md_report": (
            f"# Stock Research Report: {ticker}\n\n"
            f"> ⚠️ **Research Stopped** — {msg}\n\n"
            f"---\n\n"
            f"{_fundamentals_md(ticker, f)}"
        ),
        "cli_report": f"\n  Research Stopped: {ticker}\n  {msg}\n",
    }


def end_not_found_node(state: State):
    msg = f"{state['ticker'].upper()} does not exist in the American market."
    return {
        "md_report": f"## Research Ended\n\n**{msg}**",
        "cli_report": f"\n  Research Ended: {msg}\n",
    }


main_builder = StateGraph(State)
main_builder.add_node("run_fundamentals", run_fundamentals_node)
main_builder.add_node("run_moat_research", run_moat_research_node)
main_builder.add_node("run_risk_research", run_risk_research_node)
main_builder.add_node("aggregate", aggregate_node)
main_builder.add_node("end_low_score", end_low_score_node)
main_builder.add_node("end_not_found", end_not_found_node)

main_builder.add_edge(START, "run_fundamentals")
main_builder.add_conditional_edges(
    "run_fundamentals",
    route_after_fundamentals,
    ["end_low_score", "end_not_found", "run_moat_research", "run_risk_research"],
)
main_builder.add_edge("run_moat_research", "aggregate")
main_builder.add_edge("run_risk_research", "aggregate")
main_builder.add_edge("aggregate", END)
main_builder.add_edge("end_low_score", END)
main_builder.add_edge("end_not_found", END)
main_graph = main_builder.compile()
