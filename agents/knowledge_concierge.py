"""
Knowledge Concierge Agent - Gestiona consultas generales usando RAG.
"""
from langchain_core.messages import AIMessage
from agents.base import AgentState
from knowledge.retriever import chain as rag_chain


def knowledge_concierge_node(state: AgentState) -> AgentState:
    """
    Nodo de Conocimiento: Responde a consultas generales utilizando RAG.
    
    Args:
        state: Estado actual de la conversación
        
    Returns:
        Estado actualizado con la respuesta del agente
    """
    print("--- Ejecutando Knowledge Concierge ---")
    # Extraer la última pregunta del usuario del estado
    user_question = state["messages"][-1].content
    # Invocar la cadena RAG con la pregunta del usuario
    response = rag_chain.invoke(user_question)
    # Devolvemos la respuesta como un mensaje de IA para añadirlo al historial
    return {"messages": [AIMessage(content=response)]}
