from retriever import build_retriever

retriever = build_retriever()

def retrieve_node(state):

    docs = retriever.invoke(state["question"])

    context = [d.page_content for d in docs]

    return {"context": context}


from llm import generate_answer

def generate_node(state):
    context = "\n".join(state["context"])

    answer = generate_answer(
        state["question"],
        context
    )

    return {"answer": answer}


def decide_node(state):

    if "NO" in state["reflection"]:
        return "retrieve"

    return "end"