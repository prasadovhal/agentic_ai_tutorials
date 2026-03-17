from llm import llm

def reflection_node(state):

    prompt = f"""
Evaluate this answer.

Question:
{state['question']}

Answer:
{state['answer']}

Is the answer correct and complete?

Reply only YES or NO and explain.
"""

    result = llm.invoke(prompt)

    return {"reflection": result}