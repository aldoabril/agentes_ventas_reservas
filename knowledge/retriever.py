import os
from dotenv import load_dotenv
load_dotenv()


if not os.environ.get("LANGSMITH_API_KEY"):
    os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")


from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# Get the absolute path to the project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_DB_PATH = os.path.join(PROJECT_ROOT, "data", "chroma_langchain_db")

# Initialize embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Initialize Chroma vector store
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory=CHROMA_DB_PATH,
)


# Stage 2: Retrieval and Generation
retriever = vector_store.as_retriever()

# Prompt template
template = """Responde la pregunta basándote únicamente en el siguiente contexto:
{context}

Pregunta: {question}

Si la respuesta no se encuentra en el contexto, responde educadamente que no tienes esa información en este momento.
"""
prompt = ChatPromptTemplate.from_template(template)

# LLM
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# RAG chain
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Example usage (commented out to avoid execution on import):
# print(chain.invoke("Dame información de Clarus Dent"))
