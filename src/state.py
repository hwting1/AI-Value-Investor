from typing import Annotated, List, Optional, TypedDict
from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from .schema import FundamentalsSchema, MoatSchema, RiskSchema


class State(TypedDict):
    ticker: str
    fundamentals_result: Optional[FundamentalsSchema]
    moat_result: Optional[MoatSchema]
    risk_result: Optional[RiskSchema]
    md_report: Optional[str]
    cli_report: Optional[str]


class FundamentalsState(TypedDict):
    ticker: str
    messages: Annotated[List[AnyMessage], add_messages]
    fundamentals_result: Optional[FundamentalsSchema]


class MoatState(TypedDict):
    ticker: str
    messages: Annotated[List[AnyMessage], add_messages]
    moat_result: Optional[MoatSchema]


class RiskState(TypedDict):
    ticker: str
    messages: Annotated[List[AnyMessage], add_messages]
    risk_result: Optional[RiskSchema]
