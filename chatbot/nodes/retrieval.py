from chatbot.retriever import build_retriever

retriever = build_retriever()

def retrieval_node(state):

    queries = state.get("sub_questions") or [state["rewritten_query"]]

    contexts = []

    for q in queries:
        docs = retriever.invoke(q)
        contexts.extend([d.page_content for d in docs])

    return {"contexts": contexts}