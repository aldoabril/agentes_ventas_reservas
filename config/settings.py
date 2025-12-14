"""
Configuración centralizada del sistema de agentes de ventas y reservas.
"""
import os
from dotenv import load_dotenv
from enum import Enum
# Cargar variables de entorno
load_dotenv()

# === IDs de Empresa y Paciente ===
EMPRESA_ID = "A0OZsgiMQQVtwxMhN6Um"
PACIENTE_ID = "OcUAJ7TQGxPOsndeg30j"

# === Configuración de API ===
API_BASE_URL = "https://us-central1-odontoplus-4db47.cloudfunctions.net/api/v1/citas"
CLINIC_TIMEZONE = "America/Lima"

# === Configuración de MCP Server ===
MCP_SERVER_URL = "http://localhost:8000/mcp"
MCP_SERVER_PORT = 8000

# === Configuración de LLM ===
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "openai")

class GEMINI_MODELS(Enum):
    GEMINI_3_PRO = "gemini-3.5-pro"
    GEMINI_25_FLASH = "gemini-2.5-flash"
    GEMINI_25_FLASH_LITE = "gemini-2.5-flash-lite"

class GPT_MODELS(Enum):
    GPT_4O_MINI = "gpt-4o-mini"
    GPT_4O = "gpt-4o"
    GPT_4O_01 = "gpt-4o-01"

# === Configuración de Embeddings ===
EMBEDDING_MODEL = "text-embedding-3-large"

# === Configuración de Vector Store ===
VECTOR_STORE_COLLECTION = "example_collection"
VECTOR_STORE_PATH = "./data/chroma_langchain_db"

# === Configuración de Documentos ===
DOCUMENTS_PATH = "./knowledge/documents"

# === Especialistas ===
ESPECIALISTAS = {
    "GMcKghlgHvTkoPxj9t4X": "Dr. Juan Pérez",
    "oxFsA3phVDEVuw3jCQFx": "Dr. María López"
}

# === Configuración de RAG ===
RAG_CONFIG = {
    "top_k": 5,  # Número de documentos a recuperar
    "similarity_threshold": 0.25,  # Umbral mínimo de similitud (0.0-1.0)
    "search_type": "similarity",  # "similarity" o "mmr" (Maximum Marginal Relevance)
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "fetch_k": 20,  # Número de documentos a recuperar antes de filtrar por threshold
}

# === Configuración de LLM para RAG ===
# Modelo por defecto para RAG (puede ser diferente del orchestrator)
RAG_LLM_MODEL_OPENAI = GPT_MODELS.GPT_4O_MINI.value
RAG_LLM_MODEL_GEMINI = GEMINI_MODELS.GEMINI_25_FLASH.value

# === Configuración de Logging ===
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
LOG_FORMAT = os.environ.get("LOG_FORMAT", "standard")  # "json" o "standard"
LOG_FILE = os.environ.get("LOG_FILE", None)  # Opcional: ruta a archivo de log

# === Configuración de LangSmith ===
LANGCHAIN_TRACING_V2 = os.environ.get("LANGCHAIN_TRACING_V2", "false").lower() == "true"
LANGCHAIN_API_KEY = os.environ.get("LANGCHAIN_API_KEY") or os.environ.get("LANGSMITH_API_KEY")
LANGCHAIN_PROJECT = os.environ.get("LANGCHAIN_PROJECT", "agentes-ventas-reservas")
LANGCHAIN_ENDPOINT = os.environ.get("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")

# === Configuración de Error Handling ===
MAX_RETRY_ATTEMPTS = int(os.environ.get("MAX_RETRY_ATTEMPTS", "3"))
RETRY_INITIAL_WAIT = float(os.environ.get("RETRY_INITIAL_WAIT", "1.0"))
RETRY_MAX_WAIT = float(os.environ.get("RETRY_MAX_WAIT", "10.0"))