import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from chatbot.graph import app
from chatbot.evaluation.dataset import dataset

results = []

for row in dataset:

    output = app.invoke({
        "question": row["question"]
    })

    results.append({
        "question": row["question"],
        "answer": output["answer"],
        "metrics": output["metrics"],
        "judge": output["judge_score"]
    })

print(results)