from langchain_core.messages import HumanMessage, SystemMessage
from langchain_tavily import TavilyCrawl, TavilySearch
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from ..llm import assistant_llm, base_llm
from ..schema import MoatSchema
from ..state import MoatState
from ..system_prompts import moat_system_prompt

research_tools = [TavilySearch(), TavilyCrawl()]
research_llm = base_llm.bind_tools(research_tools)
moat_extract_llm = assistant_llm.with_structured_output(MoatSchema)


def moat_research_node(state: MoatState):
    if not state["messages"]:
        messages = [SystemMessage(content=moat_system_prompt),
                    HumanMessage(content=state["ticker"])]
        return {"messages": messages + [research_llm.invoke(messages)]}
    else:
        return {"messages": research_llm.invoke(state["messages"])}


def moat_extract_node(state: MoatState):
    messages = state["messages"] + [
        HumanMessage(content="請根據以上研究，以指定格式回傳護城河評分結果。")
    ]
    return {"moat_result": moat_extract_llm.invoke(messages)}


moat_builder = StateGraph(MoatState)
moat_builder.add_node("moat_research", moat_research_node)
moat_builder.add_node("tools", ToolNode(research_tools))
moat_builder.add_node("moat_extract", moat_extract_node)
moat_builder.add_edge(START, "moat_research")
moat_builder.add_conditional_edges(
    "moat_research", tools_condition,
    {"tools": "tools", "__end__": "moat_extract"},
)
moat_builder.add_edge("tools", "moat_research")
moat_builder.add_edge("moat_extract", END)
moat_subgraph = moat_builder.compile()
