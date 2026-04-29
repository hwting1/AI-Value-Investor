"""
ingest.py — Fetch annual fundamentals for a stock ticker and persist to DuckDB.

Data sources:
  Alpha Vantage : free_cash_flow     (CASH_FLOW endpoint)
                  dividend_per_share (DIVIDENDS endpoint)
  Finnhub       : company profile    (company_profile2)
                  eps, net_margin, lt_debt_to_equity, roic
                  (company_basic_financials → series.annual)

DB schema (auto-created on first run):
  companies                — one row per ticker
  annual_financial_metrics — one row per (company_id, fiscal_year)

Usage:
    uv run python ingest.py AAPL
    uv run python ingest.py AAPL MSFT KO
    uv run python ingest.py --db finance.duckdb AAPL
"""
from __future__ import annotations

import argparse
import datetime
import os
import time
from dataclasses import dataclass
from typing import Any

import duckdb
import finnhub
import pandas as pd
import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ── Config ────────────────────────────────────────────────────────────────────

_AV_BASE_URL = "https://www.alphavantage.co/query"
_AV_DELAY_SECONDS = 13      # Alpha Vantage free tier: 5 req / min
DEFAULT_DB = "fundamentals.duckdb"

# Global runtime config — populated from .env at startup
@dataclass
class RuntimeConfig:
    av_key: str = ""
    fh_key: str = ""
    db_path: str = DEFAULT_DB


_CONFIG = RuntimeConfig()

_FH_METRIC_SPECS: dict[str, tuple[tuple[str, ...], float]] = {
    "eps": (("epsAnnual", "eps"), 1.0),
    "net_margin": (("netMarginAnnual", "netMargin"), 100.0),
    "lt_debt_to_equity": (("longtermDebtTotalEquity",), 1.0),
    "roic": (("roicAnnual", "roic"), 100.0),
}

_METRIC_COLUMNS = {
    "eps": "eps (USD)",
    "free_cash_flow": "free_cash_flow (M USD)",
    "dividend_per_share": "dividend_per_share (USD)",
    "net_margin": "net_margin (%)",
    "lt_debt_to_equity": "lt_debt_to_equity",
    "roic": "roic (%)",
}

# ── Helpers ───────────────────────────────────────────────────────────────────


def _to_float(value: Any) -> float | None:
    if value in (None, "", "None", "null", "NaN"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _year(date_str: Any) -> int | None:
    try:
        return int(str(date_str)[:4]) if date_str else None
    except ValueError:
        return None


def _fiscal_year_window() -> tuple[int, int]:
    current_year = datetime.date.today().year
    return current_year - 10, current_year - 1


# ── Data models ───────────────────────────────────────────────────────────────


@dataclass
class CompanyProfile:
    ticker: str
    company_name: str | None
    sector: str | None
    company_url: str | None


@dataclass
class AnnualMetrics:
    fiscal_year: int
    eps: float | None = None
    free_cash_flow: float | None = None
    dividend_per_share: float | None = None
    net_margin: float | None = None
    lt_debt_to_equity: float | None = None
    roic: float | None = None


# ── Alpha Vantage ─────────────────────────────────────────────────────────────


class _AVSession:
    """HTTP session with automatic rate-limit delay for Alpha Vantage."""

    def __init__(self, api_key: str) -> None:
        self._key = api_key
        self._http = requests.Session()
        self._calls = 0

    def get(self, function: str, symbol: str) -> dict[str, Any]:
        if self._calls > 0:
            time.sleep(_AV_DELAY_SECONDS)
        self._calls += 1
        resp = self._http.get(
            _AV_BASE_URL,
            params={"function": function, "symbol": symbol, "apikey": self._key},
            timeout=30,
        )
        resp.raise_for_status()
        data: dict[str, Any] = resp.json()
        for err_key in ("Note", "Information", "Error Message"):
            if err_key in data:
                raise RuntimeError(f"Alpha Vantage [{function}/{symbol}]: {data[err_key]}")
        return data


def _av_fetch(ticker: str, av_key: str) -> tuple[dict[int, float], dict[int, float]]:
    """Return (fcf_by_year, dividend_by_year) from Alpha Vantage."""
    session = _AVSession(av_key)

    cash_flow = session.get("CASH_FLOW", ticker)
    dividends = session.get("DIVIDENDS", ticker)

    fcf: dict[int, float] = {}
    for report in cash_flow.get("annualReports", []):
        year = _year(report.get("fiscalDateEnding"))
        op_cf = _to_float(report.get("operatingCashflow"))
        capex = _to_float(report.get("capitalExpenditures"))
        if year and op_cf is not None and capex is not None:
            fcf[year] = (op_cf - capex) / 1_000_000

    div: dict[int, float] = {}
    for event in dividends.get("data", []):
        year = _year(event.get("ex_dividend_date"))
        amount = _to_float(event.get("amount"))
        if year and amount is not None:
            div[year] = div.get(year, 0.0) + amount

    return fcf, div


# ── Finnhub ───────────────────────────────────────────────────────────────────


def _fh_client(api_key: str) -> finnhub.Client:
    """Create a Finnhub SDK client with increased timeout and retry."""
    client = finnhub.Client(api_key=api_key)
    retry = Retry(total=3, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry)
    client._session.mount("https://", adapter)
    client._session.mount("http://", adapter)
    # Increase timeout: (connect_timeout, read_timeout)
    original_request = client._session.request

    def _patched_request(method, url, **kwargs):
        kwargs.setdefault("timeout", (10, 30))
        return original_request(method, url, **kwargs)

    client._session.request = _patched_request
    return client


def _fh_fetch(
    ticker: str, fh_key: str
) -> tuple[CompanyProfile, dict[int, dict[str, float | None]]]:
    """Return (CompanyProfile, {fiscal_year: metrics_dict}) from Finnhub."""
    client: finnhub.Client = _fh_client(fh_key)

    profile_raw = client.company_profile2(symbol=ticker)
    profile = CompanyProfile(
        ticker=ticker,
        company_name=profile_raw.get("name"),
        sector=profile_raw.get("finnhubIndustry"),
        company_url=profile_raw.get("weburl"),
    )

    basic = client.company_basic_financials(ticker, "all")
    annual = basic.get("series", {}).get("annual", {})

    def _year_map(*keys: str) -> dict[int, float]:
        for key in keys:
            pts = annual.get(key, [])
            if pts:
                return {
                    int(pt["period"][:4]): float(pt["v"])
                    for pt in pts
                    if pt.get("period") and pt.get("v") is not None
                }
        return {}

    metric_maps = {
        name: _year_map(*keys)
        for name, (keys, _scale) in _FH_METRIC_SPECS.items()
    }

    def _scaled_metrics_for_year(year: int) -> dict[str, float | None]:
        return {
            name: metric_maps[name][year] * scale if year in metric_maps[name] else None
            for name, (_keys, scale) in _FH_METRIC_SPECS.items()
        }

    all_years = sorted(set().union(*metric_maps.values()))
    metrics: dict[int, dict[str, float | None]] = {
        year: _scaled_metrics_for_year(year) for year in all_years
    }
    return profile, metrics


# ── Merge ─────────────────────────────────────────────────────────────────────


def _merge(
    av_fcf: dict[int, float],
    av_div: dict[int, float],
    fh_metrics: dict[int, dict[str, float | None]],
) -> list[AnnualMetrics]:
    year_min, year_max = _fiscal_year_window()

    all_years = sorted(
        y for y in (set(av_fcf) | set(av_div) | set(fh_metrics))
        if year_min <= y <= year_max
    )

    def _row_for_year(year: int) -> AnnualMetrics:
        fh = fh_metrics.get(year, {})
        return AnnualMetrics(
            fiscal_year=year,
            eps=fh.get("eps"),
            free_cash_flow=av_fcf.get(year),
            dividend_per_share=av_div.get(year),
            net_margin=fh.get("net_margin"),
            lt_debt_to_equity=fh.get("lt_debt_to_equity"),
            roic=fh.get("roic"),
        )

    return [
        _row_for_year(year) for year in all_years
    ]


# ── DuckDB ────────────────────────────────────────────────────────────────────

_DDL = """
CREATE SEQUENCE IF NOT EXISTS company_id_seq START 1;

CREATE TABLE IF NOT EXISTS companies (
    company_id   BIGINT  DEFAULT nextval('company_id_seq') PRIMARY KEY,
    ticker       VARCHAR(32) UNIQUE NOT NULL,
    company_name TEXT,
    sector       TEXT,
    company_url  TEXT,
    created_at   TIMESTAMP DEFAULT current_timestamp
);

CREATE TABLE IF NOT EXISTS annual_financial_metrics (
    company_id         BIGINT   NOT NULL REFERENCES companies(company_id),
    fiscal_year        INTEGER  NOT NULL,
    eps                DECIMAL(20,6),
    free_cash_flow     DECIMAL(20,4),   -- in millions
    dividend_per_share DECIMAL(20,6),
    net_margin         DECIMAL(20,6),
    lt_debt_to_equity  DECIMAL(20,6),
    roic               DECIMAL(20,6),
    created_at         TIMESTAMP DEFAULT current_timestamp,
    PRIMARY KEY (company_id, fiscal_year)
);
"""


def open_db(db_path: str) -> duckdb.DuckDBPyConnection:
    conn = duckdb.connect(db_path)
    for stmt in (s.strip() for s in _DDL.split(";") if s.strip()):
        conn.execute(stmt)
    return conn


def _upsert_company(conn: duckdb.DuckDBPyConnection, profile: CompanyProfile) -> int:
    conn.execute(
        """
        INSERT INTO companies (ticker, company_name, sector, company_url)
        VALUES (?, ?, ?, ?)
        ON CONFLICT (ticker) DO UPDATE SET
            company_name = excluded.company_name,
            sector       = excluded.sector,
            company_url  = excluded.company_url
        """,
        [profile.ticker, profile.company_name, profile.sector, profile.company_url],
    )
    row = conn.execute(
        "SELECT company_id FROM companies WHERE ticker = ?", [profile.ticker]
    ).fetchone()
    return int(row[0])  # type: ignore[index]


def _metric_params(company_id: int, m: AnnualMetrics) -> list[float | int | None]:
    return [
        company_id,
        m.fiscal_year,
        m.eps,
        m.free_cash_flow,
        m.dividend_per_share,
        m.net_margin,
        m.lt_debt_to_equity,
        m.roic,
    ]


def _upsert_metrics(
    conn: duckdb.DuckDBPyConnection, company_id: int, rows: list[AnnualMetrics]
) -> None:
    for m in rows:
        conn.execute(
            """
            INSERT INTO annual_financial_metrics
                (company_id, fiscal_year, eps, free_cash_flow, dividend_per_share,
                 net_margin, lt_debt_to_equity, roic)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (company_id, fiscal_year) DO UPDATE SET
                eps                = excluded.eps,
                free_cash_flow     = excluded.free_cash_flow,
                dividend_per_share = excluded.dividend_per_share,
                net_margin         = excluded.net_margin,
                lt_debt_to_equity  = excluded.lt_debt_to_equity,
                roic               = excluded.roic
            """,
            _metric_params(company_id, m),
        )


# ── Public API ────────────────────────────────────────────────────────────────


def _metrics_to_markdown(df: pd.DataFrame) -> str:
    return (
        df.rename(columns=_METRIC_COLUMNS)
        .set_index("fiscal_year")
        .to_markdown(floatfmt=".2f", intfmt="d")
    )


def query_metrics(ticker: str) -> str:
    """Query annual financial metrics for *ticker* from DuckDB.

    Returns a Markdown table of the last 10 fiscal years (current_year-10 to
    current_year-1) covering: EPS, free cash flow, dividend per share,
    net margin, LT debt-to-equity, and ROIC.
    """
    ticker = ticker.upper()
    year_min, year_max = _fiscal_year_window()

    if not os.path.exists(_CONFIG.db_path):
        return (
            f"No database found at `{_CONFIG.db_path}`. "
            f'Please call `ingest_ticker("{ticker}")` first to load the company\'s data.'
        )

    conn = duckdb.connect(_CONFIG.db_path, read_only=True)
    try:
        df: pd.DataFrame = conn.execute(
            """
            SELECT
                m.fiscal_year,
                m.eps,
                m.free_cash_flow,
                m.dividend_per_share,
                m.net_margin,
                m.lt_debt_to_equity,
                m.roic
            FROM annual_financial_metrics m
            JOIN companies c USING (company_id)
            WHERE c.ticker = ?
              AND m.fiscal_year BETWEEN ? AND ?
            ORDER BY m.fiscal_year DESC
            """,
            [ticker, year_min, year_max],
        ).df()
    finally:
        conn.close()

    if df.empty:
        return (
            f"No data found for **{ticker}**. "
            f'Please run `ingest_ticker("{ticker}")` first to load the company\'s data.'
        )

    return _metrics_to_markdown(df)


def ingest_ticker(ticker: str) -> str:
    """Fetch annual fundamentals for *ticker* from Alpha Vantage + Finnhub and
    upsert into DuckDB.

    Data sources:
      Alpha Vantage  ->  free_cash_flow, dividend_per_share
      Finnhub        ->  company profile, eps, net_margin, lt_debt_to_equity, roic

    On success, returns a Markdown table of the ingested data (via query_metrics).
    On failure (ticker not found or no data within range), returns a plain-text
    message explaining the issue.

    API keys and DB path are read from runtime config populated by
    load_config().
    """
    ticker = ticker.upper()
    conn = open_db(_CONFIG.db_path)
    try:
        print(f"[{ticker}] Fetching Finnhub (profile + metrics)...")
        profile, fh_metrics = _fh_fetch(ticker, _CONFIG.fh_key)

        if not profile.company_name:
            return (
                f"[{ticker}] Ticker not found — Finnhub returned no profile data. "
                "Please verify the ticker symbol is correct."
            )

        print(f"[{ticker}] Fetching Alpha Vantage (CASH_FLOW + DIVIDENDS)...")
        av_fcf, av_div = _av_fetch(ticker, _CONFIG.av_key)

        rows = _merge(av_fcf, av_div, fh_metrics)
        if not rows:
            return (
                f"[{ticker}] No annual metrics found — the APIs returned data for this "
                "ticker but none fell within the expected date range. "
                "The company may be too new or have insufficient history."
            )

        company_id = _upsert_company(conn, profile)
        _upsert_metrics(conn, company_id, rows)
    finally:
        conn.close()

    return query_metrics(ticker)


# ── CLI ───────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Ingest stock fundamentals into DuckDB from Alpha Vantage + Finnhub."
    )
    p.add_argument("symbols", nargs="+", help="Ticker symbols, e.g. AAPL MSFT KO")
    p.add_argument(
        "--db",
        default=DEFAULT_DB,
        help=f"DuckDB file path (default: {DEFAULT_DB})",
    )
    return p.parse_args()


def load_config(db_path: str | None = None) -> None:
    """Load API keys from .env into runtime config and set the DB path."""
    load_dotenv()
    av_key = os.getenv("ALPHA_VANTAGE_KEY", "")
    fh_key = os.getenv("FINNHUB_KEY", "")
    missing = [n for n, v in [("ALPHA_VANTAGE_KEY", av_key), ("FINNHUB_KEY", fh_key)] if not v]
    if missing:
        raise RuntimeError(f"Missing .env variable(s): {', '.join(missing)}")
    _CONFIG.av_key = av_key
    _CONFIG.fh_key = fh_key
    if db_path:
        _CONFIG.db_path = db_path


def main() -> None:
    args = _parse_args()
    load_config(db_path=args.db)
    print(f"Database: {_CONFIG.db_path}\n")

    for i, sym in enumerate(args.symbols):
        if i > 0:
            # Respect Alpha Vantage rate limit between tickers
            print(f"Waiting {_AV_DELAY_SECONDS}s before next ticker...")
            time.sleep(_AV_DELAY_SECONDS)
        print(ingest_ticker(sym))

    print("\nAll done.")


if __name__ == "__main__":
    main()
