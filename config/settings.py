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
