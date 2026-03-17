from langchain_community.llms import Ollama

llm = Ollama(model="mistral")

def generate_answer(question, context):

    prompt = f"""
Use the context to answer the question.

Context:
{context}

Question:
{question}

Answer clearly.
"""

    return llm.invoke(prompt)