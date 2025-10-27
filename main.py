import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage
from agents import (
    AgentState,
    lead_qualifier_node,
    knowledge_concierge_node,
    scheduler_node,
    negotiator_node,
    guardian_node,
    router_node
)

# --- 1. Definición del Grafo de Agentes ---

# Crear una instancia del grafo de estado
workflow = StateGraph(AgentState)

# Añadir los nodos al grafo. Cada nodo es una función que hemos definido en agents.py
workflow.add_node("Lead Qualifier", lead_qualifier_node)
workflow.add_node("Orchestrator", router_node)
workflow.add_node("Knowledge Concierge", knowledge_concierge_node)
workflow.add_node("Scheduler", scheduler_node)
workflow.add_node("Negotiator", negotiator_node)
workflow.add_node("Guardian", guardian_node)

# --- 2. Definición de las Transiciones (Edges) ---

# Establecer el punto de entrada del grafo
workflow.set_entry_point("Lead Qualifier")

# Conectar el Lead Qualifier con el Orchestrator
workflow.add_edge("Lead Qualifier", "Orchestrator")

# Definir la lógica de enrutamiento condicional desde el Orchestrator
def decide_next_node(state: AgentState):
    """Función que lee el estado y decide el siguiente paso."""
    return state["next_node"]

workflow.add_conditional_edges(
    "Orchestrator",
    decide_next_node,
    {
        "Knowledge Concierge": "Knowledge Concierge",
        "Scheduler": "Scheduler",
        "Negotiator": "Negotiator",
        "Guardian": "Guardian",
        # Si el router decidiera finalizar, podríamos añadir:
        # "end": END
    }
)

# Por ahora, todos los agentes especializados finalizan el flujo
workflow.add_edge("Knowledge Concierge", END)
workflow.add_edge("Scheduler", END)
workflow.add_edge("Negotiator", END)
workflow.add_edge("Guardian", END)

# --- 3. Compilación y Ejecución Conversacional ---

if __name__ == "__main__":
    load_dotenv()

    # Instanciar el MemorySaver para mantener el estado entre turnos
    memory = MemorySaver()

    # Compilar el grafo en una aplicación ejecutable
    app = workflow.compile(checkpointer=memory)

    # Configurar un ID de hilo para la conversación
    thread_id = "mi-conversacion-1"
    config = {"configurable": {"thread_id": thread_id}}

    print("🤖 Hola, soy tu asistente. Escribe 'salir' para terminar.")
    while True:
        user_input = input("🙂 Tú: ")
        if user_input.lower() in ["salir", "exit"]:
            break
        
        # El input ahora es una lista de mensajes que se añadirán al estado
        inputs = {"messages": [HumanMessage(content=user_input)]}
        
        # El stream guardará y cargará automáticamente el estado para el thread_id
        for event in app.stream(inputs, config, stream_mode="values"):
            event["messages"][-1].pretty_print()
