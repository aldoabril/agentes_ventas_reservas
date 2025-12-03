import os
from dotenv import load_dotenv

load_dotenv()

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.utils import filter_complex_metadata

# Get the absolute path to the project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_DB_PATH = os.path.join(PROJECT_ROOT, "data", "chroma_langchain_db")
DOCUMENTS_PATH = os.path.join(os.path.dirname(__file__), "documents")

# Initialize embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Initialize Chroma vector store
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory=CHROMA_DB_PATH,
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

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200
            )
            splits = text_splitter.split_documents(docs)
            filtered_splits = filter_complex_metadata(splits)
            vector_store.add_documents(filtered_splits)
            print(f"Loaded and split {filename}")
        except Exception as e:
            print(f"Failed to process {filename}: {e}")


def load_markdown_from_directory(directory_path):
    """Loads all Markdown files from a directory, splits them into documents, and adds them to the vector store."""
    md_files = [
        f for f in os.listdir(directory_path) if f.endswith((".md", ".markdown"))
    ]
    if not md_files:
        print(f"No Markdown files found in {directory_path}.")
        return

    for filename in md_files:
        file_path = os.path.join(directory_path, filename)
        try:
            loader = UnstructuredMarkdownLoader(file_path, mode="elements")
            docs = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200
            )
            splits = text_splitter.split_documents(docs)
            filtered_splits = filter_complex_metadata(splits)
            vector_store.add_documents(filtered_splits)
            print(f"Loaded and split {filename}")
        except Exception as e:
            print(f"Failed to process {filename}: {e}")


# Example usage:
# The documents directory is in the same folder as this file
if not os.path.exists(DOCUMENTS_PATH):
    os.makedirs(DOCUMENTS_PATH)
    print(
        f"Created directory: {DOCUMENTS_PATH}. Please add your PDF and Markdown files here."
    )

load_pdfs_from_directory(DOCUMENTS_PATH)
# load_markdown_from_directory(DOCUMENTS_PATH)
