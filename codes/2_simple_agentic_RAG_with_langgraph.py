import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from constant import huggingface_api_key, GOOGLE_API_KEY
from langchain_google_genai import ChatGoogleGenerativeAI

os.environ["HUGGINGFACEHUB_API_TOKEN"] = huggingface_api_key
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

############################################################################
## Load data

from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_PATH = "D:/Study/Git_repo/agentic_ai_tutorials/codes/docs/"

loader = DirectoryLoader(
    DATA_PATH,
    glob="*.txt",
    loader_cls=TextLoader
)

docs = loader.load()

print(len(docs))

############################################################################
# create chunks

splitter = RecursiveCharacterTextSplitter(chunk_size=500)

chunks = splitter.split_documents(docs)

############################################################################
## create embeddings

from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

############################################################################
## vector database

from langchain_community.vectorstores import FAISS

vectorstore = FAISS.from_documents(chunks, embeddings)

retriever = vectorstore.as_retriever()

############################################################################

# LLM model

from langchain_community.llms import Ollama
llm = Ollama(model="mistral")

############################################################################
## Tool creation
"""
Agent needs tools.
    Tool 1 → Retriever
    Tool 2 → Calculator
    Tool 3 → Web Search
"""

## Retriever tool

from langchain_core.tools import tool


@tool
def retrieve_docs(query):
    """Retrieve EV related documents."""
    docs = retriever.invoke(query)
    return "\n".join([d.page_content for d in docs])


## Calculator Tool

@tool
def calculator(expr):
    """Evaluate mathematical expressions."""
    return eval(expr)


## Web search tool

from duckduckgo_search import DDGS

@tool
def web_search(query):
    """Search internet for latest information."""
    results = []

    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=5):
            results.append(r["body"])

    return "\n".join(results)


## Tool List

tools = [
    retrieve_docs,
    calculator,
    web_search
]

############################################################################

# LangGraph Agentic RAG : Minimal

from langgraph.graph import StateGraph

def agent_node(state):

    query = state["query"]

    decision = llm.invoke(
        f"Does this question require document retrieval?\n{query}"
    )

    if "yes" in str(decision).lower():
        return {
            "query": query,
            "route": "retrieve"
        }

    return {
        "query": query,
        "route": "answer"
    }


def retrieve_node(state):

    query = state["query"]

    context = retrieve_docs.invoke(query)

    return {
        "context": context
    }
    

def answer_node(state):

    query = state["query"]
    context = state.get("context", "")

    answer = llm.invoke(
        f"""
Context:
{context}

Question:
{query}
"""
    )

    return {
        "answer": str(answer)
    }

from langgraph.graph import StateGraph, START, END
from typing import TypedDict

from typing import TypedDict, Optional

class State(TypedDict, total=False):
    query: str
    context: str
    answer: str
    route: str


graph = StateGraph(State)

graph.add_node("agent", agent_node)
graph.add_node("retrieve", retrieve_node)
graph.add_node("answer", answer_node)


# ENTRY POINT
graph.add_edge(START, "agent")

# Conditional Routing (Core Agent Logic)

graph.add_conditional_edges(
    "agent",
    lambda state: state["route"],
    {
        "retrieve": "retrieve",
        "answer": "answer"
    }
)
# FLOW
graph.add_edge("retrieve", "answer")

# end point
graph.add_edge("answer", END)


app = graph.compile()


result = app.invoke({
    "query": "Compare Tesla and BYD revenue"
})

print(result["answer"])


"""
Internal Execution Flow: LangGraph internally runs:

START
 ↓
agent_node()
 ↓
retrieve_node()
 ↓
answer_node()
 ↓
END

"""


############################################################################

