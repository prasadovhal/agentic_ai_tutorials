def compute_metrics(state):

    metrics = {}

    metrics["context_count"] = len(state["contexts"])
    metrics["answer_length"] = len(state["answer"])

    # simple hallucination proxy
    metrics["grounded"] = "YES" in state["reflection"]

    return {"metrics": metrics}