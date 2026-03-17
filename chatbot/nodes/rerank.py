from sentence_transformers import CrossEncoder

reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def rerank_node(state):

    pairs = [(state["question"], c) for c in state["contexts"]]

    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(state["contexts"], scores),
        key=lambda x: x[1],
        reverse=True
    )

    top_contexts = [c for c, _ in ranked[:3]]

    return {"contexts": top_contexts}