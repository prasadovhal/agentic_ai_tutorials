import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langgraph.graph import StateGraph, END

from chatbot.state import AgentState
from chatbot.nodes.reflection import reflection_node
from chatbot.nodes.rewrite import rewrite_node
from chatbot.nodes.self_ask import self_ask_node
from chatbot.nodes.retrieval import retrieval_node
from chatbot.nodes.rerank import rerank_node
from chatbot.nodes.generate import generate_node
from chatbot.nodes.judge import judge_node
from chatbot.evaluation.metrics import compute_metrics
from chatbot.memory.memory import memory_node

workflow = StateGraph(AgentState)

workflow.add_node("rewrite", rewrite_node)
workflow.add_node("self_ask", self_ask_node)
workflow.add_node("retrieve", retrieval_node)
workflow.add_node("rerank", rerank_node)
workflow.add_node("generate", generate_node)
workflow.add_node("reflect", reflection_node)
workflow.add_node("judge", judge_node)
workflow.add_node("metrics", compute_metrics)
workflow.add_node("memory", memory_node)

workflow.set_entry_point("rewrite")

workflow.add_edge("rewrite", "self_ask")
workflow.add_edge("self_ask", "retrieve")
workflow.add_edge("retrieve", "rerank")
workflow.add_edge("rerank", "generate")
workflow.add_edge("generate", "reflect")

workflow.add_conditional_edges(
    "reflect",
    lambda state: "retry" if "NO" in state["reflection"] else "continue",
    {
        "retry": "retrieve",
        "continue": "judge"
    }
)

workflow.add_edge("judge", "metrics")
workflow.add_edge("metrics", "memory")
workflow.add_edge("memory", END)

app = workflow.compile()