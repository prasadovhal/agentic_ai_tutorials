from chatbot.llm import llm

def rewrite_node(state):

    prompt = f"""
Rewrite the query for better retrieval:

Query: {state['question']}
"""

    rewritten = llm.invoke(prompt)

    return {"rewritten_query": rewritten}