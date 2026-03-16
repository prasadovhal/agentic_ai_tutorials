import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from constant import huggingface_api_key, GOOGLE_API_KEY
from langchain_google_genai import ChatGoogleGenerativeAI

os.environ["HUGGINGFACEHUB_API_TOKEN"] = huggingface_api_key
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

##################################################################################

# load data

import pandas as pd

DATA_PATH = "D:/Study/Git_repo/agentic_ai_tutorials/codes/docs/"
df = pd.read_csv(DATA_PATH + "rag_llm_judge_eval_dataset.csv")
df.drop(["id"],axis=1,inplace=True)

dataset = []

for _, row in df.iterrows():

    dataset.append({
        "question": row["question"],
        "ground_truth": row["ground_truth"],
        "contexts": row["context"],
        "answer": row["generated_answer"]
    })
    
##################################################################################

# Create LLM Judge Prompt

judge_prompt = """
You are an expert AI evaluator.

Question:
{question}

Context:
{context}

Answer:
{answer}

Ground Truth:
{ground_truth}

Score between 0 and 1:

1. Answer Relevance
2. Faithfulness
3. Correctness
4. Hallucination

Return JSON only:

{{
 "answer_relevance": float,
 "faithfulness": float,
 "correctness": float,
 "hallucination": float
}}
"""

# implement judge llm

import ollama
import json

def judge_llm(question, context, answer, ground_truth):

    prompt = judge_prompt.format(
        question=question,
        context=context,
        answer=answer,
        ground_truth=ground_truth
    )

    response = ollama.chat(
        model="mistral",
        messages=[{"role":"user","content":prompt}],
        options={"temperature":0}
    )

    text = response["message"]["content"]

    return json.loads(text)

# run loop

scores = []

for sample in dataset:

    result = judge_llm(
        sample["question"],
        sample["contexts"],
        sample["answer"],
        sample["ground_truth"]
    )

    scores.append(result)
    

# Aggregate Scores

df = pd.DataFrame(scores)

metrics = {
    "avg_relevance": df["answer_relevance"].mean(),
    "avg_faithfulness": df["faithfulness"].mean(),
    "avg_correctness": df["correctness"].mean(),
    "avg_hallucination": df["hallucination"].mean()
}

print(metrics)