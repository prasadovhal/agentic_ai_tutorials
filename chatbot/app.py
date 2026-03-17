import streamlit as st
from graph import app

st.title("Agentic RAG Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

user_input = st.chat_input("Ask anything")

if user_input:

    st.session_state.messages.append(
        {"role":"user","content":user_input}
    )

    result = app.invoke({
        "question": user_input
    })

    answer = result["answer"]

    st.session_state.messages.append(
        {"role":"assistant","content":answer}
    )

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):
        st.write(msg["content"])