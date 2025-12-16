"""
Workflow de conversación - Define el grafo de estados para el sistema multi-agente.
"""
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from agents import (
    AgentState,
    router_node,
    knowledge_concierge_node,
    scheduler_node,
    lead_qualifier_node,
    guardian_node,
    human_handoff_node,
)


def decide_next_node(state: AgentState):
    """
    Función que lee el estado y decide el siguiente paso.
    
    Args:
        state: Estado actual de la conversación
        
    Returns:
        Nombre del siguiente nodo
    """
    return state["next_node"]


def create_app():
    """
    Crea y compila el grafo de conversación.
    
    Returns:
        Aplicación compilada de LangGraph
    """
    # Crear una instancia del grafo de estado
    workflow = StateGraph(AgentState)
    
    # Añadir los nodos al grafo
    workflow.add_node("Lead Qualifier", lead_qualifier_node)
    workflow.add_node("Orchestrator", router_node)
    workflow.add_node("Knowledge Concierge", knowledge_concierge_node)
    workflow.add_node("Scheduler", scheduler_node)
    workflow.add_node("Guardian", guardian_node)
    workflow.add_node("Human Handoff", human_handoff_node)
    
    # Establecer el punto de entrada del grafo
    workflow.set_entry_point("Lead Qualifier")

    # Definir la lógica de enrutamiento condicional desde el Lead Qualifier
    workflow.add_conditional_edges(
        "Lead Qualifier",
        decide_next_node,
        {
            "Orchestrator": "Orchestrator",
            "end": END
        }
    )
    
    # Definir la lógica de enrutamiento condicional desde el Orchestrator
    workflow.add_conditional_edges(
        "Orchestrator",
        decide_next_node,
        {
            "Knowledge Concierge": "Knowledge Concierge",
            "Scheduler": "Scheduler",
            "end": END
        }
    )
    
    # Definir edges desde los agentes especializados al Guardian (intercept before END)
    # The Guardian validates all actions before they reach the user
    workflow.add_edge("Knowledge Concierge", "Guardian")
    workflow.add_edge("Scheduler", "Guardian")
    
    # Guardian routes based on verdict
    workflow.add_conditional_edges(
        "Guardian",
        decide_next_node,
        {
            "Knowledge Concierge": "Knowledge Concierge",  # Retry with feedback
            "Scheduler": "Scheduler",  # Retry with feedback
            "Human Handoff": "Human Handoff",  # Escalate
            "end": END  # Approved
        }
    )
    
    # Human Handoff always ends
    workflow.add_edge("Human Handoff", END)
    
    # Compilar el grafo con memoria
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    
    return app
