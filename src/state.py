from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages

from .schema import FundamentalsSchema, MoatSchema, RiskSchema


class State(TypedDict):
    ticker: str
    fundamentals_result: FundamentalsSchema | None
    moat_result: MoatSchema | None
    risk_result: RiskSchema | None
    md_report: str | None
    cli_report: str | None


class FundamentalsState(TypedDict):
    ticker: str
    messages: Annotated[list[AnyMessage], add_messages]
    fundamentals_result: FundamentalsSchema | None


class MoatState(TypedDict):
    ticker: str
    messages: Annotated[list[AnyMessage], add_messages]
    moat_result: MoatSchema | None


class RiskState(TypedDict):
    ticker: str
    messages: Annotated[list[AnyMessage], add_messages]
    risk_result: RiskSchema | None
