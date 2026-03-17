def memory_node(state):

    history = state.get("chat_history", [])

    history.append(state["question"])
    history.append(state["answer"])

    return {"chat_history": history}