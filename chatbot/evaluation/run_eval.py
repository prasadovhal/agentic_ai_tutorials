from graph import app
from dataset import dataset

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