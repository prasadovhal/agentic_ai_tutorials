# agentic_ai_tutorials
Understanding Agentic AI and its framework hands on with python

## Set up Python & Poetry

1. cd agentic_ai_tutorials
2. install poetry
`(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -`
3. run `C:\Users\user_name\AppData\Roaming\Python\Scripts`
4. check poetry version `poetry --version`
5. set `poetry config virtualenvs.in-project true`
6. run `poetry install`
7. set venv 
   - for windows `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned` or `.venv\Scripts\activate`
   - for linux/mac `source .venv/bin/activate`

## changes you need to make

1. Create `constant.py` in main folder.
2. Add the following keys inside it:
   - `GOOGLE_API_KEY = "your_google_api_key"`
   - `OPENAI_KEY = "your_openai_key"`
   - `HUGGINGFACE_API_KEY = "your_huggingface_api_key"`
   - `LANGFUSE_SECRET_KEY = "your_langfuse_secret_key"`
   - `LANGFUSE_PUBLIC_KEY = "your_langfuse_public_key"`
   - `LANGFUSE_BASE_URL = "https://cloud.langfuse.com"`

# Roadmap
| LEVEL | TOPIC | KEY HANDS-ON TASK |
| :--- | :--- | :--- |
| **Beginner** | Tool Calling | Bind a `get_weather` function to an LLM using Pydantic. |
| **Intermediate** | ReAct Framework | Build a terminal agent that can use a Calculator and Wikipedia. |
| **Intermediate** | Memory Systems | Implement a "Summary Buffer" memory that condenses old chats. |
| **Advanced** | Agentic RAG | Build a RAG system that "self-corrects" if context is poor. |
| **Expert** | GraphRAG | Convert a PDF into a Neo4j graph and query it via Cypher. |
| **Expert** | Multi-Agent | Use LangGraph to make two agents "debate" a topic. |

# Topics included

1. Basics of Agent and its parts
    - Memory
    - Tools
    - Action
    - Tool calling agent
    - Reasoning
    - ReAct Agent
    - Planning
2. Simple Agentic RAG
    - Create tools, 
    - How RAG is converted to Agentic RAG
3. LangGraph
    - Simple Agentic RAG
    - Full Agentic RAG
    - Relection
4. Multi Agent Systems
    - using autogen
    - using crewai
    - using llamaindex
    - using haystack
5. Agentic RAG with Self-Ask
6. Evaluations
7. Kaggle Competitions
    - LLM Science Exam competition
8. Create your own chatbot with Agentic AI and streamlit