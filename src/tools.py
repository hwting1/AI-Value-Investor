from langchain_core.tools import StructuredTool

from ingest import ingest_ticker, query_metrics

ingest_ticker_tool = StructuredTool.from_function(
    func=ingest_ticker,
    name="ingest_ticker",
    description=(
        "Fetch annual fundamentals for a stock ticker from Alpha Vantage and Finnhub "
        "and persist to DuckDB. Call this when the database has no data for the ticker yet. "
        "Returns a Markdown table on success, or an error message."
    ),
)

query_metrics_tool = StructuredTool.from_function(
    func=query_metrics,
    name="query_metrics",
    description=(
        "Query the last 10 fiscal years of annual financial metrics for a stock ticker "
        "from DuckDB. Returns a Markdown table. If no data exists, instructs to run ingest first."
    ),
)


