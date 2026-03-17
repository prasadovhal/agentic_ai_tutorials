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

train = train.sample(200)

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

# define llm: Ollama LLM Function

import ollama

def mistral_llm(prompt):

    response = ollama.chat(
        model="mistral",
        messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"]


#########################################################################

# create agents

## Self-Ask Agent : Break question into sub-questions.

def self_ask(question):

    prompt = f"""
Break the question into subquestions.

Question:
{question}

Return list.
"""

    return mistral_llm(prompt)

## Reasoning Agent

def reasoning_agent(question, contexts, choices):

    context_text = "\n".join(contexts)

    prompt = f"""
Answer MCQ.

Context:
{context_text}

Question:
{question}

Choices:
{choices}

Return final answer text only.
"""

    return mistral_llm(prompt)

## Reflection Agent : Checks if answer is reliable.

def reflection_agent(question, answer, contexts):

    prompt = f"""
Evaluate answer correctness.

Question:
{question}

Answer:
{answer}

Context:
{contexts}

Is answer supported by context?
Return YES or NO.
"""

    return mistral_llm(prompt)

#########################################################################

# Full Agentic Pipeline

def agentic_rag_pipeline(
    question,
    choices,
    corpus,
    faiss_index,
    bm25,
    tokenized_corpus,
    embeddings
):

    sub_questions = self_ask(question)

    docs = hybrid_retrieve(
        question,
        corpus,
        faiss_index,
        bm25,
        tokenized_corpus,
        embeddings
    )

    docs = rerank(question, docs)

    answer = reasoning_agent(
        question,
        docs,
        choices
    )

    reflection = reflection_agent(
        question,
        answer,
        docs
    )

    return {
        "question": question,
        "answer": answer,
        "contexts": docs,
        "reflection": reflection
    }
    
#########################################################################

# LLM-as-Judge Evaluation

def judge_llm(answer, ground_truth, contexts):

    prompt = f"""
Evaluate answer.

Ground Truth:
{ground_truth}

Answer:
{answer}

Context:
{contexts}

Return JSON:

{{
 "answer_relevance": 0-1,
 "faithfulness": 0-1,
 "context_relevance": 0-1
}}
"""

    return mistral_llm(prompt)

## RAG Observability Metrics

### Retrieval Recall

def retrieval_recall(contexts, ground_truth):

    for c in contexts:

        if ground_truth.lower() in c.lower():
            return 1

    return 0

### Faithfulness

def faithfulness(answer, contexts):

    text = " ".join(contexts)

    if answer.lower() in text.lower():
        return 1
    else:
        return 0
    
### Context Precision

def context_precision(contexts, ground_truth):

    relevant = 0

    for c in contexts:

        if ground_truth.lower() in c.lower():
            relevant += 1

    return relevant / len(contexts)

#########################################################################

# run evaluation

results = []

for row in gold_dataset:

    result = agentic_rag_pipeline(
        row["question"],
        row["choices"],
        corpus,
        faiss_index,
        bm25,
        tokenized_corpus,
        embeddings
    )

    judge = judge_llm(
        result["answer"],
        row["ground_truth"],
        result["contexts"]
    )

    recall = retrieval_recall(
        result["contexts"],
        row["ground_truth"]
    )

    results.append({
        "question": row["question"],
        "answer": result["answer"],
        "judge": judge,
        "retrieval_recall": recall
    })
    
op = pd.DataFrame(results)

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

judge_metrics = op["judge"].apply(lambda x: extract_judge_metrics(x))
#########################################################################

# submission file

submission = []

for r in results:

    submission.append({
        "answer": r["answer"]
    })

submission_df = pd.DataFrame(submission)

submission_df.to_csv(
    "submission.csv",
    index=False
)