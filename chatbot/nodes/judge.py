from chatbot.llm import llm

def judge_node(state):

    prompt = f"""
Evaluate answer quality:

Question: {state['question']}
Answer: {state['answer']}

Score:
- relevance (0-1)
- groundedness (0-1)
- correctness (0-1)

Return JSON.
"""

    result = llm.invoke(prompt)

    return {"judge_score": result}