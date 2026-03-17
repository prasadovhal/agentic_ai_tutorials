def self_ask_node(state):

    prompt = f"""
Break the question into sub-questions if needed.

Question:
{state['rewritten_query']}

Return list.
"""

    result = llm.invoke(prompt)

    sub_questions = result.split("\n")

    return {"sub_questions": sub_questions}