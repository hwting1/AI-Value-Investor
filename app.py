import streamlit as st
from agent import main_graph

st.set_page_config(page_title="AI Value Investor", page_icon="📈", layout="wide")

st.title("📈 AI Value Investor")
st.caption("Analyse a US-listed stock using fundamental data, moat research, and risk assessment.")

# ── Input ──────────────────────────────────────────────────────────────────────
with st.form("ticker_form"):
    ticker = st.text_input("Stock Ticker", placeholder="e.g. AAPL, MSFT, KO").strip().upper()
    submitted = st.form_submit_button("Analyse", type="primary")

# ── Run ────────────────────────────────────────────────────────────────────────
if submitted:
    if not ticker:
        st.warning("Please enter a ticker symbol.")
        st.stop()

    progress = st.status(f"Researching **{ticker}**…", expanded=True)

    with progress:
        step_fundamentals = st.empty()
        step_moat         = st.empty()
        step_risk         = st.empty()
        step_report       = st.empty()

        step_fundamentals.write("⏳ Fetching fundamentals…")

        result: dict = {}
        for event in main_graph.stream({"ticker": ticker}, stream_mode="updates"):
            node = next(iter(event))
            data = event[node]

            if node == "run_fundamentals":
                step_fundamentals.write("✅ Fundamentals done")
                f = data.get("fundamentals_result")
                if f and not f.stock_exists:
                    progress.update(label="Ticker not found.", state="error", expanded=False)
                    st.error(f"**{ticker}** does not exist in the American market.")
                    st.stop()
                if f and f.total_score < 4:
                    progress.update(label=f"Low score ({f.total_score:.1f}/6). Research stopped.", state="complete", expanded=False)

            elif node in ("run_moat_research", "moat_research"):
                step_moat.write("⏳ Researching moat…")
            elif node in ("run_risk_research", "risk_research"):
                step_risk.write("⏳ Researching risks…")
            elif node == "aggregate":
                step_moat.write("✅ Moat done")
                step_risk.write("✅ Risk done")
                step_report.write("✅ Report generated")

            result.update(data)

        progress.update(label="Analysis complete!", state="complete", expanded=False)

    # ── Display ────────────────────────────────────────────────────────────────
    md_report  = result.get("md_report", "")
    cli_report = result.get("cli_report", "")

    st.markdown(md_report)
