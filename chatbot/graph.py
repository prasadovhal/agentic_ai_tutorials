from langgraph.graph import StateGraph, END

from state import AgentState
from nodes import retrieve_node, generate_node, decide_node
from reflection import reflection_node

workflow = StateGraph(AgentState)

workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)
workflow.add_node("reflect", reflection_node)

workflow.set_entry_point("retrieve")

workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", "reflect")

workflow.add_conditional_edges(
    "reflect",
    decide_node,
    {
        "retrieve": "retrieve",
        "end": END
    }
)

app = workflow.compile()