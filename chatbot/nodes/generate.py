import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from chatbot.llm import llm

def generate_node(state):

    context = "\n".join(state["contexts"])

    prompt = f"""
Answer using context.

Context:
{context}

Question:
{state['question']}
"""

    answer = llm.invoke(prompt)

    return {"answer": answer}