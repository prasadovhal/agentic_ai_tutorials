import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from constant import huggingface_api_key

os.environ["HUGGINGFACEHUB_API_TOKEN"] = huggingface_api_key

##########################################################################

import pandas as pd

DATA_PATH = "D:/Study/Git_repo/agentic_ai_tutorials/competitions/LLM_Science_Exam/data/"
train = pd.read_csv(DATA_PATH + "train.csv")
test = pd.read_csv(DATA_PATH + "test.csv")

train = train.sample(50)

#train.head()

train.drop(['id'],axis=1,inplace=True)
test.drop(['id'],axis=1,inplace=True)

# wikipedia Knowledge corpus 

wiki_corpus = pd.read_csv(DATA_PATH + "wiki_stem_corpus.csv")
wiki_corpus.head()
wiki_corpus = wiki_corpus[["content_id", "page_title", "text"]]
wiki_corpus = wiki_corpus.sample(2000)


documents = []

for _, row in wiki_corpus.iterrows():
    doc = {
        "id": row["content_id"],
        "title": row["page_title"],
        "text": row["text"]
    }
    documents.append(doc)
    
#########################################################################

# gold dataset

def build_gold_dataset(df):

    dataset = []

    for _, row in df.iterrows():

        question = row["prompt"]

        choices = [
            row["A"],
            row["B"],
            row["C"],
            row["D"],
            row["E"]
        ]

        answer = choices[ord(row["answer"]) - 65]

        dataset.append({
            "question": question,
            "choices": choices,
            "ground_truth": answer
        })

    return dataset


gold_dataset = build_gold_dataset(train)

#########################################################################
# Adaptive Chunking: Token budget chunking

def adaptive_chunk(text, max_tokens=200):

    sentences = text.split(". ")

    chunks = []
    current = ""

    for s in sentences:

        if len(current) + len(s) < max_tokens:
            current += s + ". "
        else:
            chunks.append(current)
            current = s

    chunks.append(current)

    return chunks


corpus = []

for article in documents:

    chunks = adaptive_chunk(article["text"])

    for c in chunks:

        corpus.append({
            "text": c,
            "title": article["title"]
        })
        
#########################################################################

from sentence_transformers import SentenceTransformer
import numpy as np

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

def create_embeddings(corpus):

    texts = [c["text"] for c in corpus]

    embeddings = embedding_model.encode(
        texts,
        show_progress_bar=True
    )

    return np.array(embeddings)

embeddings = create_embeddings(corpus)

#########################################################################

# Build FAISS Index

import faiss

def build_faiss_index(embeddings):

    dim = embeddings.shape[1]

    index = faiss.IndexFlatL2(dim)

    index.add(embeddings)

    return index

faiss_index = build_faiss_index(embeddings)

###########################

# BM25 Index
from rank_bm25 import BM25Okapi

def build_bm25(corpus):

    tokenized = [
        c["text"].split()
        for c in corpus
    ]

    bm25 = BM25Okapi(tokenized)

    return bm25, tokenized

bm25, tokenized_corpus = build_bm25(corpus)

###########################

# hybrid retreival

def hybrid_retrieve(
    query,
    corpus,
    faiss_index,
    bm25,
    tokenized_corpus,
    embeddings,
    top_k=5
):

    query_vec = embedding_model.encode([query])

    _, dense_ids = faiss_index.search(query_vec, top_k)

    bm25_scores = bm25.get_scores(query.split())

    bm25_ids = np.argsort(bm25_scores)[-top_k:]

    ids = set(dense_ids[0]) | set(bm25_ids)

    docs = [corpus[i]["text"] for i in ids]

    return docs

#########################################################################

# Re-ranker

from sentence_transformers import CrossEncoder

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

def rerank(query, docs, top_k=3):

    pairs = [(query, d) for d in docs]

    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(docs, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return [d for d, _ in ranked[:top_k]]

#########################################################################


# Ollama LLM Wrapper

import ollama

def mistral_llm(prompt):

    response = ollama.chat(
        model="mistral",
        messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"]

############################################################################

# Define Graph State

from typing import TypedDict, List

class AgentState(TypedDict):

    question: str
    choices: List[str]

    sub_questions: List[str]

    contexts: List[str]

    answer: str

    reflection: str

    judge: dict
    
###########################

# Query Rewrite Node

def query_rewrite_node(state):

    question = state["question"]

    prompt = f"""
Rewrite the question to improve search retrieval.

Question:
{question}
"""

    new_query = mistral_llm(prompt)

    return {"question": new_query}

###########################

# Self-Ask Node : Agent decomposes problem.

def self_ask_node(state):

    question = state["question"]

    prompt = f"""
Break the question into smaller subquestions.

Question:
{question}

Return list.
"""

    sub_questions = mistral_llm(prompt)

    return {"sub_questions": sub_questions}

###########################

# Reasoning Node : Uses hybrid retrieval.

def retrieval_node(state):

    question = state["question"]

    docs = hybrid_retrieve(
        question,
        corpus,
        faiss_index,
        bm25,
        tokenized_corpus,
        embeddings
    )

    docs = rerank(question, docs)

    return {"contexts": docs}

###########################

# Reasoning Node : LLM generates answer.

def reasoning_node(state):

    question = state["question"]
    contexts = state["contexts"]
    choices = state["choices"]

    context_text = "\n".join(contexts)

    prompt = f"""
Use context to answer MCQ.

Context:
{context_text}

Question:
{question}

Choices:
{choices}

Return correct answer.
"""

    answer = mistral_llm(prompt)

    return {"answer": answer}

###########################

# Reflection Node : Detect hallucination.

def reflection_node(state):

    question = state["question"]
    answer = state["answer"]
    contexts = state["contexts"]

    prompt = f"""
Check if the answer is supported by the context.

Question:
{question}

Answer:
{answer}

Context:
{contexts}

Return YES or NO.
"""

    result = mistral_llm(prompt)

    return {"reflection": result}

###########################

# Conditional Retry Logic: If reflection fails → retrieve again.

def reflection_router(state):

    if "NO" in state["reflection"]:
        return "retrieve_again"

    return "judge"

###########################

# LLM as Judge Node : Evaluation stage.

import json

def judge_node(state):

    answer = state["answer"]
    contexts = state["contexts"]

    prompt = f"""
Evaluate answer quality.

Answer:
{answer}

Context:
{contexts}

Return JSON:

{{
    "answer_relevance": 1-5,
    "faithfulness": 1-5,
    "context_relevance": 1-5
}}
"""

    response = mistral_llm(prompt)

    try:
        judge = json.loads(response)
    except:
        judge = {}

    return {"judge": judge}

############################################################################

# Build langgraph

from langgraph.graph import StateGraph, START, END

## define nodes
workflow = StateGraph(AgentState)

workflow.add_node("rewrite", query_rewrite_node)
workflow.add_node("self_ask", self_ask_node)
workflow.add_node("retrieve", retrieval_node)
workflow.add_node("reason", reasoning_node)
workflow.add_node("reflect", reflection_node)
workflow.add_node("judge", judge_node)

## define edges

workflow.add_edge(START, "rewrite")
workflow.add_edge("rewrite", "self_ask")
workflow.add_edge("self_ask", "retrieve")
workflow.add_edge("retrieve", "reason")
workflow.add_edge("reason", "reflect")

## conditional edge

workflow.add_conditional_edges(
    "reflect",
    reflection_router,
    {
        "retrieve_again": "retrieve",
        "judge": "judge"
    }
)

## end graph

workflow.add_edge("judge", END)

## compile
graph = workflow.compile()

############################################################################

## Run Agentic Pipeline

for _, row in train.iterrows():

    question = row["prompt"]

    choices = [
        row["A"],
        row["B"],
        row["C"],
        row["D"],
        row["E"]
    ]

    result = graph.invoke({
        "question": question,
        "choices": choices
    })

    print(result)

############################################################################

# RAG Observability Metrics

## Retrieval Recall
def retrieval_recall(contexts, ground_truth):

    for c in contexts:
        if ground_truth.lower() in c.lower():
            return 1

    return 0


## Faithfulness

def faithfulness(answer, contexts):

    context = " ".join(contexts)

    return int(answer.lower() in context.lower())


## context precision

def context_precision(contexts, ground_truth):

    relevant = 0

    for c in contexts:
        if ground_truth.lower() in c.lower():
            relevant += 1

    return relevant / len(contexts)


def extract_judge_metrics(judge_dict):

    # Default values
    metrics = {
        "answer_relevance": 0,
        "faithfulness_llm": 0,
        "context_relevance": 0
    }

    if not isinstance(judge_dict, dict):
        return metrics

    metrics["answer_relevance"] = judge_dict.get("answer_relevance", 0)
    metrics["faithfulness_llm"] = judge_dict.get("faithfulness", 0)
    metrics["context_relevance"] = judge_dict.get("context_relevance", 0)

    return metrics


## evalution loop

results = []

for row in gold_dataset:

    result = graph.invoke({
        "question": row["question"],
        "choices": row["choices"]
    })

    # Observability metrics
    recall = retrieval_recall(
        result["contexts"],
        row["ground_truth"]
    )

    faith = faithfulness(
        result["answer"],
        result["contexts"]
    )

    precision = context_precision(
        result["contexts"],
        row["ground_truth"]
    )

    # Extract judge metrics
    judge_metrics = extract_judge_metrics(result["judge"])

    results.append({
        "question": row["question"],
        "answer": result["answer"],

        # Retrieval metrics
        "retrieval_recall": recall,
        "context_precision": precision,

        # Faithfulness
        "faithfulness_score": faith,

        # LLM judge metrics
        "answer_relevance": judge_metrics["answer_relevance"],
        "faithfulness_llm": judge_metrics["faithfulness_llm"],
        "context_relevance": judge_metrics["context_relevance"]
    })
    

results_df = pd.DataFrame(results)

summary = {

    "avg_retrieval_recall": results_df["retrieval_recall"].mean(),

    "avg_context_precision": results_df["context_precision"].mean(),

    "avg_faithfulness": results_df["faithfulness_score"].mean(),

    "avg_answer_relevance": results_df["answer_relevance"].mean(),

    "avg_llm_faithfulness": results_df["faithfulness_llm"].mean(),

    "avg_context_relevance": results_df["context_relevance"].mean()
}

print(summary)