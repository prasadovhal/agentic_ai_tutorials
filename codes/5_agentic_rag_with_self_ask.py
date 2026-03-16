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

# FULL AGENTIC RAG WITH LANGGRAPH

from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END

# LLM (example using Ollama)
from langchain_community.llms import Ollama
llm = Ollama(model="mistral")


# -------------------------------
# STATE SCHEMA
# -------------------------------

class State(TypedDict, total=False):
    query: str
    sub_questions: List[str]  # for self ask
    sub_answers: List[str]    # for self ask
    context: str
    answer: str

# -------------------------------
# TOOLS
# -------------------------------

from langchain_core.tools import tool

@tool
def retrieve_docs(query):
    """Retrieve EV related documents."""
    docs = retriever.invoke(query)
    return "\n".join([d for d in docs])


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


def retrieve_tool(query):
    docs = retrieve_docs.invoke(query)
    return docs


def web_search_tool(query):
    return web_search.invoke(query)


def calculator_tool(expr):
    return calculator.invoke(expr)


TOOLS = {
    "retrieve": retrieve_tool,
    "search": web_search_tool,
    "calculator": calculator_tool
}


# -------------------------------
# AGENT NODE (REASONING)
# -------------------------------

def agent_node(state):

    query = state["query"]

    prompt = f"""
You are an AI agent.

Available tools:
retrieve - search EV documents
search - search the internet
calculator - perform math

Decide the next step.

Question: {query}

Respond in format:
Thought: reasoning
Action: retrieve | search | calculator | finish
Action Input: input
"""

    response = llm.invoke(prompt)

    response = str(response)

    thought = ""
    action = ""
    action_input = ""

    for line in response.split("\n"):
        if "Thought:" in line:
            thought = line.split("Thought:")[1].strip()
        if "Action:" in line:
            action = line.split("Action:")[1].strip()
        if "Action Input:" in line:
            action_input = line.split("Action Input:")[1].strip()

    return {
        "thought": thought,
        "action": action,
        "query": action_input if action_input else query
    }


# -------------------------------
# TOOL EXECUTION NODE
# -------------------------------

def tool_node(state):

    action = state["action"]
    query = state["query"]

    if action == "finish":
        return {}

    if action in TOOLS:
        observation = TOOLS[action](query)
    else:
        observation = "Unknown tool"

    return {
        "observation": observation,
        "context": observation
    }


# -------------------------------
# ANSWER NODE
# -------------------------------

def answer_node(state):

    query = state["query"]
    context = state.get("context", "")

    prompt = f"""
Use the following information to answer.

Context:
{context}

Question:
{query}
"""

    answer = llm.invoke(prompt)

    return {
        "answer": str(answer)
    }


# -------------------------------
# REFLECTION NODE
# -------------------------------

def reflection_node(state):

    query = state["query"]
    answer = state["answer"]

    critique = llm.invoke(f"""
Check if this answer correctly answers the question.

Question:
{query}

Answer:
{answer}

Respond GOOD or BAD.
""")

    critique = str(critique).lower()

    if "bad" in critique:
        return {"action": "retrieve"}

    return {}


# -------------------------------
# Self-Ask Node
# -------------------------------

def self_ask_node(state):

    query = state["query"]

    prompt = f"""
Break the question into smaller sub-questions.

Question:
{query}

Return as list.
"""

    response = llm.invoke(prompt)

    questions = str(response).split("\n")

    return {
        "sub_questions": questions
    }

# -------------------------------
# Sub-Question Retrieval Node
# -------------------------------

def subqa_node(state):

    answers = []

    for q in state["sub_questions"]:

        docs = retrieve_docs.invoke(q)

        context = "\n".join([d for d in docs])

        response = llm.invoke(
            f"""
Context:
{context}

Question:
{q}
"""
        )

        answers.append(str(response))

    return {
        "sub_answers": answers
    }

# -------------------------------
# Final Synthesis Node
# -------------------------------

def synthesis_node(state):

    query = state["query"]
    answers = state["sub_answers"]

    prompt = f"""
Use the answers below to answer the original question.

Original Question:
{query}

Sub Answers:
{answers}
"""

    final_answer = llm.invoke(prompt)

    return {
        "answer": str(final_answer)
    }

# -------------------------------
# BUILD GRAPH
# -------------------------------

from langgraph.graph import StateGraph, START, END

graph = StateGraph(State)

graph.add_node("self_ask", self_ask_node)
graph.add_node("subqa", subqa_node)
graph.add_node("synthesis", synthesis_node)

graph.add_edge(START, "self_ask")
graph.add_edge("self_ask", "subqa")
graph.add_edge("subqa", "synthesis")
graph.add_edge("synthesis", END)

app = graph.compile()

# -------------------------------
# RUN
# -------------------------------

result = app.invoke({
    "query": "Compare Tesla and BYD revenue"
})

print(result["answer"])