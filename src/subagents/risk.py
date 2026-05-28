from langchain_core.messages import HumanMessage, SystemMessage
from langchain_tavily import TavilyCrawl, TavilySearch
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from ..llm import assistant_llm, base_llm
from ..schema import RiskSchema
from ..state import RiskState
from ..system_prompts import risk_system_prompt

research_tools = [TavilySearch(), TavilyCrawl()]
research_llm = base_llm.bind_tools(research_tools)
risk_extract_llm = assistant_llm.with_structured_output(RiskSchema)


def risk_research_node(state: RiskState):
    if not state["messages"]:
        messages = [SystemMessage(content=risk_system_prompt),
                    HumanMessage(content=state["ticker"])]
        return {"messages": messages + [research_llm.invoke(messages)]}
    else:
        return {"messages": research_llm.invoke(state["messages"])}


def risk_extract_node(state: RiskState):
    messages = state["messages"] + [
        HumanMessage(content="請根據以上研究，以指定格式回傳風險評分結果。")
    ]
    return {"risk_result": risk_extract_llm.invoke(messages)}


risk_builder = StateGraph(RiskState)
risk_builder.add_node("risk_research", risk_research_node)
risk_builder.add_node("tools", ToolNode(research_tools))
risk_builder.add_node("risk_extract", risk_extract_node)
risk_builder.add_edge(START, "risk_research")
risk_builder.add_conditional_edges(
    "risk_research", tools_condition,
    {"tools": "tools", "__end__": "risk_extract"},
)
risk_builder.add_edge("tools", "risk_research")
risk_builder.add_edge("risk_extract", END)
risk_subgraph = risk_builder.compile()
