import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from constant import huggingface_api_key, GOOGLE_API_KEY, openai_key
from langchain_google_genai import ChatGoogleGenerativeAI

os.environ["HUGGINGFACEHUB_API_TOKEN"] = huggingface_api_key
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
os.environ["OPENAI_API_KEY"] = openai_key

############################################################################
## Load data

from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_PATH = "D:/Study/Git_repo/agentic_ai_tutorials/codes/docs/"

loader = DirectoryLoader(
    DATA_PATH,
    glob="*.txt",
    loader_cls=TextLoader
)

docs = loader.load()

print(len(docs))

############################################################################
# create chunks

splitter = RecursiveCharacterTextSplitter(chunk_size=500)

chunks = splitter.split_documents(docs)

############################################################################
## create embeddings

from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

############################################################################
## vector database

from langchain_community.vectorstores import FAISS

vectorstore = FAISS.from_documents(chunks, embeddings)

retriever = vectorstore.as_retriever()

############################################################################
## RAG query

query = "Compare Tesla and BYD revenue"

docs = retriever.invoke(query)

context = "\n".join([d.page_content for d in docs])

############################################################################
## generate answer

from langchain_community.llms import Ollama
llm = Ollama(model="mistral")

############################################################################

# Multi Agent system

##########################################
# using autogen

from autogen import AssistantAgent, UserProxyAgent

code_execution_config = {"use_docker": False}

llm_config = {
    "config_list": [
        {
            "model": "mistral",
            "base_url": "http://localhost:11434/v1",
            "api_key": "ollama",
            "price": [0, 0]
        }
    ]
}

assistant = AssistantAgent(
    name="research_agent",
    llm_config=llm_config,
    system_message="""
You are a research assistant.

Answer questions directly using your knowledge.
Do NOT generate Python scripts unless explicitly asked.
Provide concise factual answers.
"""
)

user_proxy = UserProxyAgent(
    name="user",
    human_input_mode="NEVER",
    code_execution_config=False
)

user_proxy.initiate_chat(
    assistant,
    message="Compare Tesla vs BYD revenue"
)



###############################################################

# using crewai

from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o")

research_agent = Agent(
    role="Researcher",
    goal="Retrieve EV industry data",
    backstory="Expert in electric vehicle companies and financial analysis.",
    llm=llm
)

task = Task(
    description="Compare Tesla vs BYD revenue",
    expected_output="A clear comparison of Tesla and BYD revenue.",
    agent=research_agent
)

crew = Crew(
    agents=[research_agent],
    tasks=[task]
)

result = crew.kickoff()

print(result)

##############################################################

# using llamaindex

from llama_index.core import VectorStoreIndex
from llama_index.core import SimpleDirectoryReader

# load documents
documents = SimpleDirectoryReader("docs").load_data()

# create index
index = VectorStoreIndex.from_documents(documents)

# query engine
query_engine = index.as_query_engine()

response = query_engine.query("Compare Tesla and BYD revenue")

print(response)


##############################################################

# using haystack

from haystack import Pipeline, Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.components.builders import PromptBuilder
from haystack.components.generators import OpenAIGenerator

## convert langchain Document to Haystack Document

haystack_docs = [
    Document(
        content=doc.page_content,
        meta=doc.metadata
    )
    for doc in docs
]

document_store = InMemoryDocumentStore()
document_store.write_documents(haystack_docs)

retriever = InMemoryBM25Retriever(document_store=document_store)

generator = OpenAIGenerator(model="gpt-4o-mini")


template = """
Use the following documents to answer the question.

Documents:
{% for doc in documents %}
{{ doc.content }}
{% endfor %}

Question: {{ query }}

Answer:
"""

prompt_builder = PromptBuilder(template=template)

pipeline = Pipeline()
pipeline.add_component("retriever", retriever)
pipeline.add_component("prompt_builder", prompt_builder)
pipeline.add_component("generator", generator)

pipeline.connect("retriever.documents", "prompt_builder.documents")
pipeline.connect("prompt_builder.prompt", "generator.prompt")

result = pipeline.run(
    {
        "retriever": {"query": "Compare Tesla and BYD revenue"},
        "prompt_builder": {"query": "Compare Tesla and BYD revenue"}
    }
)

print(result["generator"]["replies"][0])