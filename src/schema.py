from typing import Literal

from pydantic import BaseModel, Field


class FundamentalsSchema(BaseModel):
    stock_exists: bool = Field(default=True, description="股票是否存在於美國市場")
    eps_score: int = Field(default=0, description="EPS 評分", ge=0, le=1)
    eps_history: list[float] = Field(default_factory=list, description="EPS 過去 10 年的數據")
    free_cashflow_score: int = Field(default=0, description="Free Cash Flow 評分", ge=0, le=1)
    free_cashflow_history: list[float] = Field(default_factory=list, description="Free Cash Flow 過去 10 年的數據")
    dividend_score: int = Field(default=0, description="Dividend 評分", ge=0, le=1)
    dividend_history: list[float] = Field(default_factory=list, description="Dividend 過去 10 年的數據")
    net_margin_score: float = Field(description="Net Margin 評分", ge=0, le=1)
    net_margin_history: list[float] = Field(default_factory=list, description="Net Margin 過去 10 年的數據")
    debt_equity_score: Literal[0, 0.5, 1] = Field(default=0, description="Long Term Debt / Equity 評分")
    debt_equity_history: float = Field(default=0.0, description="最新一年的 Long Term Debt / Equity 數據")
    roic_score: int = Field(default=0, description="ROIC 評分", ge=0, le=1)
    roic_history: list[float] = Field(default_factory=list, description="ROIC 過去 10 年的數據")
    total_score: float = Field(default=0.0, description="總財務評分", ge=0, le=6)
    explaination: str = Field(default="", description="對所有項目的評分理由")


class MoatSchema(BaseModel):
    score: Literal[0, 2, 3, 4] = Field(description="護城河評分")

    keywords: list[
        Literal["無形資產", "成本優勢", "網路效應", "高轉換成本", "利基市場", "自信"]
    ] = Field(description="公司擁有護城河的類別")

    explain: dict[
        Literal["無形資產", "成本優勢", "網路效應", "高轉換成本", "利基市場", "自信"],
        str,
    ] = Field(description="每個護城河類別對應的詳細解釋")

    source: list[str] = Field(description="參考的資料來源網址")


class RiskSchema(BaseModel):
    score: int = Field(description="風險評分", ge=-3, le=0)

    keywords: list[
        Literal["政策風險", "科技風險", "關鍵人物風險"]
    ] = Field(description="公司擁有的風險類別")

    explain: dict[
        Literal["政策風險", "科技風險", "關鍵人物風險"],
        str,
    ] = Field(description="每個風險類別對應的詳細解釋")

    source: list[str] = Field(description="參考的資料來源網址")
