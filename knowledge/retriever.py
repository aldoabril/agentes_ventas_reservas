"""
Sistema RAG mejorado para Knowledge Concierge.
Incluye configuración flexible, métricas y validación de relevancia.
"""
import os
from typing import List, Tuple, Optional
from dotenv import load_dotenv
load_dotenv()


# Configurar LANGSMITH_API_KEY si no está en os.environ pero sí en .env
if not os.environ.get("LANGSMITH_API_KEY"):
    langsmith_key = os.getenv("LANGSMITH_API_KEY")
    if langsmith_key:
        os.environ["LANGSMITH_API_KEY"] = langsmith_key


from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from config import (
    RAG_CONFIG,
    EMBEDDING_MODEL,
    VECTOR_STORE_COLLECTION,
    VECTOR_STORE_PATH,
    LLM_PROVIDER,
    GPT_MODELS,
    GEMINI_MODELS,
    RAG_LLM_MODEL_OPENAI,
    RAG_LLM_MODEL_GEMINI,
)

# Importar Gemini si está disponible
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# Get the absolute path to the project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_DB_PATH = os.path.join(PROJECT_ROOT, "data", "chroma_langchain_db")

# Initialize embeddings
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

# Initialize Chroma vector store
vector_store = Chroma(
    collection_name=VECTOR_STORE_COLLECTION,
    embedding_function=embeddings,
    persist_directory=CHROMA_DB_PATH,
)


def create_retriever(
    top_k: Optional[int] = None,
    search_type: str = "similarity",
    score_threshold: Optional[float] = None,
) -> Chroma:
    """
    Crea un retriever configurable con parámetros personalizados.
    
    Args:
        top_k: Número de documentos a recuperar (default: RAG_CONFIG["top_k"])
        search_type: Tipo de búsqueda ("similarity" o "mmr")
        score_threshold: Umbral mínimo de similitud (default: RAG_CONFIG["similarity_threshold"])
        
    Returns:
        Retriever configurado
    """
    top_k = top_k or RAG_CONFIG["top_k"]
    score_threshold = score_threshold or RAG_CONFIG["similarity_threshold"]
    
    search_kwargs = {
        "k": top_k,
    }
    
    # Si usamos MMR, necesitamos fetch_k
    if search_type == "mmr":
        search_kwargs["fetch_k"] = RAG_CONFIG.get("fetch_k", top_k * 4)
    
    retriever = vector_store.as_retriever(
        search_type=search_type,
        search_kwargs=search_kwargs,
    )
    
    return retriever


def retrieve_with_scores(
    query: str,
    top_k: Optional[int] = None,
    score_threshold: Optional[float] = None,
) -> Tuple[List[Document], List[float]]:
    """
    Recupera documentos con sus scores de similitud.
    Usa threshold adaptativo: si no encuentra documentos con el threshold inicial,
    reduce automáticamente el threshold para asegurar resultados relevantes.
    
    Args:
        query: Consulta del usuario
        top_k: Número de documentos a recuperar
        score_threshold: Umbral mínimo de similitud
        
    Returns:
        Tupla de (documentos, scores)
    """
    top_k = top_k or RAG_CONFIG["top_k"]
    initial_threshold = score_threshold or RAG_CONFIG["similarity_threshold"]
    
    # Usar similarity_search_with_score para obtener scores
    docs_with_scores = vector_store.similarity_search_with_score(
        query,
        k=top_k,
    )
    
    if not docs_with_scores:
        return [], []
    
    # Separar documentos y scores, filtrar por threshold inicial
    documents = []
    scores = []
    
    for doc, score in docs_with_scores:
        # ChromaDB devuelve distancias (menor es mejor), convertir a similitud
        # Para embeddings de OpenAI con ChromaDB, la distancia típicamente está entre 0-2
        similarity = max(0.0, 1.0 - (score / 2.0))
        
        if similarity >= initial_threshold:
            documents.append(doc)
            scores.append(similarity)
    
    # Threshold adaptativo: si no encontramos documentos con el threshold inicial,
    # usar un threshold más bajo para asegurar resultados relevantes
    if len(documents) == 0:
        # Threshold mínimo adaptativo: 0.3 o el mejor score encontrado, lo que sea menor
        # Esto asegura que siempre devolvamos resultados si hay documentos disponibles
        min_threshold = 0.3
        
        # Re-filtrar con threshold más bajo
        for doc, score in docs_with_scores:
            similarity = max(0.0, 1.0 - (score / 2.0))
            if similarity >= min_threshold:
                documents.append(doc)
                scores.append(similarity)
        
        # Si aún no hay resultados (muy raro), devolver al menos el mejor documento
        if len(documents) == 0:
            best_doc, best_score = docs_with_scores[0]
            best_similarity = max(0.0, 1.0 - (best_score / 2.0))
            documents.append(best_doc)
            scores.append(best_similarity)
            print(f"   ⚠️ Usando threshold adaptativo: devolviendo mejor documento (similarity={best_similarity:.3f})")
        else:
            print(f"   ⚠️ Threshold adaptativo activado: {len(documents)} documentos encontrados con threshold {min_threshold:.2f} (inicial: {initial_threshold:.2f})")
    
    return documents, scores


# Retriever por defecto (para compatibilidad con código existente)
retriever = create_retriever()


def create_llm():
    """
    Crea el LLM apropiado según la configuración.
    
    Returns:
        Instancia del LLM configurado
    """
    if LLM_PROVIDER == "gemini" and GEMINI_AVAILABLE:
        return ChatGoogleGenerativeAI(
            model=RAG_LLM_MODEL_GEMINI,
            temperature=0,
            max_tokens=None,
            timeout=None,
        )
    else:
        return ChatOpenAI(
            model=RAG_LLM_MODEL_OPENAI,
            temperature=0,
        )


# LLM para RAG
llm = create_llm()

# Prompt template base (será reemplazado por prompts estructurados desde knowledge_concierge)
template = """Responde la pregunta basándote únicamente en el siguiente contexto:
{context}

Pregunta: {question}

Si la respuesta no se encuentra en el contexto, responde educadamente que no tienes esa información en este momento.
"""
prompt = ChatPromptTemplate.from_template(template)

# RAG chain básico (para compatibilidad)
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


def create_rag_chain_with_prompt(custom_prompt_template: str):
    """
    Crea una cadena RAG con un prompt personalizado.
    
    Args:
        custom_prompt_template: Template de prompt personalizado
        
    Returns:
        Cadena RAG configurada
    """
    custom_prompt = ChatPromptTemplate.from_template(custom_prompt_template)
    custom_retriever = create_retriever()
    
    chain = (
        {"context": custom_retriever, "question": RunnablePassthrough()}
        | custom_prompt
        | llm
        | StrOutputParser()
    )
    
    return chain
