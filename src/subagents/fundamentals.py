from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from ..llm import assistant_llm, base_llm
from ..schema import FundamentalsSchema
from ..state import FundamentalsState
from ..system_prompts import fundamentals_system_prompt
from ..tools import ingest_ticker_tool, query_metrics_tool

fundamentals_tools = [ingest_ticker_tool, query_metrics_tool]
fundamentals_llm = base_llm.bind_tools(fundamentals_tools)
fundamentals_extract_llm = assistant_llm.with_structured_output(FundamentalsSchema)


def fundamentals_scoring_node(state: FundamentalsState):
    if not state["messages"]:
        messages = [SystemMessage(content=fundamentals_system_prompt),
                    HumanMessage(content=state["ticker"])]
        return {"messages": messages + [fundamentals_llm.invoke(messages)]}
    else:
        return {"messages": fundamentals_llm.invoke(state["messages"])}


def fundamentals_extract_node(state: FundamentalsState):
    messages = state["messages"] + [
        HumanMessage(content="請根據以上研究，以指定格式回傳財務評分結果。")
    ]
    return {"fundamentals_result": fundamentals_extract_llm.invoke(messages)}


fundamentals_builder = StateGraph(FundamentalsState)
fundamentals_builder.add_node("fundamentals_scoring", fundamentals_scoring_node)
fundamentals_builder.add_node("tools", ToolNode(fundamentals_tools))
fundamentals_builder.add_node("fundamentals_extract", fundamentals_extract_node)
fundamentals_builder.add_edge(START, "fundamentals_scoring")
fundamentals_builder.add_conditional_edges(
    "fundamentals_scoring", tools_condition,
    {"tools": "tools", "__end__": "fundamentals_extract"},
)
fundamentals_builder.add_edge("tools", "fundamentals_scoring")
fundamentals_builder.add_edge("fundamentals_extract", END)
fundamentals_subgraph = fundamentals_builder.compile()
