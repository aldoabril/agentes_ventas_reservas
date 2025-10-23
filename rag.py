import os
import dotenv
dotenv.load_dotenv()

if not os.environ.get("OPENAI_API_KEY"):
  os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Initialize embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Initialize Chroma vector store
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",
)

def load_pdfs_from_directory(directory_path):
    """Loads all PDF files from a directory, splits them into documents, and adds them to the vector store."""
    pdf_files = [f for f in os.listdir(directory_path) if f.endswith(".pdf")]
    if not pdf_files:
        print(f"No PDF files found in {directory_path}.")
        return

    for filename in pdf_files:
        file_path = os.path.join(directory_path, filename)
        try:
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(docs)
            vector_store.add_documents(splits)
            print(f"Loaded and split {filename}")
        except Exception as e:
            print(f"Failed to process {filename}: {e}")

# Example usage:
# Create a directory named 'data' and place your PDF files inside it.
data_directory = "./data" 
if not os.path.exists(data_directory):
    os.makedirs(data_directory)
    print(f"Created directory: {data_directory}. Please add your PDF files here.")

load_pdfs_from_directory(data_directory)


if not os.environ.get("LANGSMITH_API_KEY"):
    os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")