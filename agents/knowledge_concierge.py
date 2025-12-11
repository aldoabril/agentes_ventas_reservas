"""
Knowledge Concierge Agent - Gestiona consultas generales usando RAG mejorado.
Incluye contexto conversacional, validación de relevancia y prompts estructurados.
"""
import time
from typing import Optional
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from agents.base import AgentState
from agents.memory import get_safe_history_window
from knowledge.retriever import (
    retrieve_with_scores,
    create_llm,
    create_retriever,
    RAG_CONFIG,
)
from knowledge.prompts import get_prompt_for_query_type
from config import LLM_PROVIDER


def extract_conversation_context(
    messages: list[BaseMessage], max_messages: int = 5
) -> str:
    """
    Extrae el contexto conversacional de los últimos mensajes.
    
    Args:
        messages: Lista de mensajes del historial
        max_messages: Número máximo de mensajes a incluir
        
    Returns:
        String formateado con el historial conversacional
    """
    if not messages:
        return "No hay historial previo."
    
    # Obtener ventana segura de mensajes
    context_messages = get_safe_history_window(messages, max_messages=max_messages)
    
    # Formatear mensajes
    formatted = []
    for msg in context_messages:
        if isinstance(msg, HumanMessage):
            formatted.append(f"Usuario: {msg.content}")
        elif isinstance(msg, AIMessage):
            formatted.append(f"Asistente: {msg.content}")
    
    if not formatted:
        return "No hay historial previo."
    
    return "\n".join(formatted)


def classify_query_type(query: str) -> str:
    """
    Clasifica el tipo de consulta para usar el prompt apropiado.
    Implementación simple basada en palabras clave.
    
    Args:
        query: Consulta del usuario
        
    Returns:
        Tipo de consulta: "price", "service", "general", "general_ambiguous"
    """
    query_lower = query.lower()
    
    # Palabras clave para precios
    price_keywords = ["precio", "costo", "cuánto", "tarifa", "pago", "pag", "dólar", "soles"]
    if any(keyword in query_lower for keyword in price_keywords):
        return "price"
    
    # Palabras clave para servicios
    service_keywords = ["servicio", "tratamiento", "procedimiento", "implante", "endodoncia", 
                        "ortodoncia", "blanqueamiento", "corona", "prótesis"]
    if any(keyword in query_lower for keyword in service_keywords):
        return "service"
    
    # Consultas muy cortas o ambiguas
    if len(query.split()) <= 3:
        return "general_ambiguous"
    
    return "general"


def format_context_from_documents(documents: list, scores: Optional[list] = None) -> str:
    """
    Formatea los documentos recuperados en un string de contexto.
    
    Args:
        documents: Lista de documentos recuperados
        scores: Lista opcional de scores de similitud
        
    Returns:
        String formateado con el contexto
    """
    if not documents:
        return "No se encontró información relevante en la base de conocimiento."
    
    context_parts = []
    for i, doc in enumerate(documents):
        content = doc.page_content.strip()
        if scores and i < len(scores):
            # Incluir score para debugging (opcional, puede removerse en producción)
            context_parts.append(f"[Documento {i+1} - Relevancia: {scores[i]:.2f}]\n{content}")
        else:
            context_parts.append(f"[Documento {i+1}]\n{content}")
    
    return "\n\n".join(context_parts)


def knowledge_concierge_node(state: AgentState) -> AgentState:
    """
    Nodo Knowledge Concierge mejorado:
    - Extrae contexto conversacional
    - Clasifica tipo de consulta
    - Recupera documentos con validación de relevancia
    - Genera respuesta con prompt estructurado
    - Maneja casos edge (sin resultados, baja confianza)
    
    Args:
        state: Estado actual de la conversación
        
    Returns:
        Estado actualizado con la respuesta del agente
    """
    start_time = time.time()
    print("--- Ejecutando Knowledge Concierge (Mejorado) ---")
    
    try:
        # 1. Extraer consulta del usuario
        messages = state.get("messages", [])
        if not messages:
            return {"messages": [AIMessage(content="No recibí ningún mensaje. ¿En qué puedo ayudarte?")]}
        
        last_message = messages[-1]
        if isinstance(last_message, HumanMessage):
            user_question = last_message.content
        else:
            user_question = str(last_message.content)
        
        if not user_question or not user_question.strip():
            return {"messages": [AIMessage(content="No pude entender tu consulta. ¿Podrías reformularla?")]}
        
        print(f"Consulta recibida: {user_question[:100]}...")
        
        # 2. Extraer contexto conversacional
        conversation_history = extract_conversation_context(messages, max_messages=5)
        print(f"Contexto extraído: {len(get_safe_history_window(messages, max_messages=5))} mensajes")
        
        # 3. Clasificar tipo de consulta
        query_type = classify_query_type(user_question)
        print(f"Tipo de consulta detectado: {query_type}")
        
        # 4. Recuperar documentos con scores
        documents, scores = retrieve_with_scores(
            user_question,
            top_k=RAG_CONFIG["top_k"],
            score_threshold=RAG_CONFIG["similarity_threshold"],
        )
        
        print(f"Documentos recuperados: {len(documents)}")
        if scores:
            avg_score = sum(scores) / len(scores) if scores else 0.0
            print(f"Score promedio de relevancia: {avg_score:.2f}")
        
        # 5. Validar relevancia y manejar casos edge
        if not documents or (scores and all(s < RAG_CONFIG["similarity_threshold"] for s in scores)):
            print("--- No se encontraron documentos relevantes ---")
            # Usar prompt para casos sin información
            prompt_template = get_prompt_for_query_type("no_info")
            prompt = ChatPromptTemplate.from_template(prompt_template)
            
            llm = create_llm()
            chain = prompt | llm | StrOutputParser()
            
            response = chain.invoke({
                "question": user_question,
                "conversation_history": conversation_history,
            })
            
            response_time = time.time() - start_time
            print(f"Tiempo de respuesta: {response_time:.2f}s")
            return {"messages": [AIMessage(content=response)]}
        
        # 6. Formatear contexto de documentos
        context = format_context_from_documents(documents, scores)
        
        # 7. Obtener prompt apropiado según tipo de consulta
        prompt_template = get_prompt_for_query_type(query_type)
        prompt = ChatPromptTemplate.from_template(prompt_template)
        
        # 8. Crear y ejecutar cadena RAG
        llm = create_llm()
        chain = (
            {
                "context": RunnablePassthrough(),
                "question": RunnablePassthrough(),
                "conversation_history": RunnablePassthrough(),
            }
            | prompt
            | llm
            | StrOutputParser()
        )
        
        response = chain.invoke({
            "context": context,
            "question": user_question,
            "conversation_history": conversation_history,
        })
        
        # 9. Validar respuesta generada
        if not response or len(response.strip()) < 10:
            print("--- Advertencia: Respuesta muy corta, regenerando ---")
            # Intentar con prompt más directo
            fallback_prompt = ChatPromptTemplate.from_template(
                "Basándote en este contexto: {context}\n\nResponde a esta pregunta: {question}\n\nResponde de manera clara y completa."
            )
            fallback_chain = fallback_prompt | llm | StrOutputParser()
            response = fallback_chain.invoke({
                "context": context,
                "question": user_question,
            })
        
        response_time = time.time() - start_time
        print(f"Respuesta generada exitosamente")
        print(f"Tiempo total de respuesta: {response_time:.2f}s")
        
        return {"messages": [AIMessage(content=response)]}
        
    except Exception as e:
        print(f"--- Error en Knowledge Concierge: {e} ---")
        import traceback
        traceback.print_exc()
        
        # Respuesta de fallback
        error_response = (
            "Lo siento, tuve un problema al procesar tu consulta. "
            "¿Podrías intentar reformularla o contactarnos directamente? "
            "También puedes agendar una cita para una consulta personalizada."
        )
        return {"messages": [AIMessage(content=error_response)]}
