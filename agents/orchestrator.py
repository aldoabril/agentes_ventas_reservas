"""
Orchestrator Agent - Router que decide el siguiente paso en la conversación.
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from agents.base import AgentState, Intent, RouteQuery


def router_node(state: AgentState) -> dict:
    """
    Nodo Router: Decide el siguiente paso basado en la intención.
    Clasifica la intención si no existe y luego enruta al agente apropiado.
    
    Args:
        state: Estado actual de la conversación
        
    Returns:
        Diccionario con next_node e intention actualizados
    """
    print("--- Ejecutando Orchestrator (Router) ---")

    # 1. Si no hay intención en el estado (o es el inicio), la clasificamos primero
    if "intention" not in state or not state["intention"]:
        print("--- Clasificando Intención (dentro del Router) ---")
        llm_classifier = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        structured_llm_classifier = llm_classifier.with_structured_output(Intent)
        
        qualifier_prompt = ChatPromptTemplate.from_messages([
            ("system", """Eres un clasificador de intenciones para un asistente de consultorio dental.
Tu tarea es analizar el mensaje del usuario y clasificarlo en una de las siguientes categorías:
- consulta: El usuario pide información (precios, horarios, servicios).
- reserva: El usuario quiere agendar una nueva cita.
- reprogramacion: El usuario quiere cambiar una cita existente.
- cancelacion: El usuario quiere cancelar una cita.
- objecion: El usuario presenta una queja o duda sobre el precio o servicio.
- invalido: El mensaje es spam, no se entiende o no está relacionado con el consultorio."""),
            ("human", "Analiza el siguiente mensaje del usuario: '{message}'"),
        ])
        
        chain_classifier = qualifier_prompt | structured_llm_classifier
        # Analizamos el último mensaje del usuario
        last_message = state["messages"][-1].content
        result_classifier = chain_classifier.invoke({"message": last_message})
        state['intention'] = result_classifier.intention
        print(f"Intención detectada: {state['intention']}")

    # 2. Configuración del LLM para tomar la decisión de enrutamiento
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    structured_llm = llm.with_structured_output(RouteQuery)

    # Prompt para el LLM
    router_prompt = ChatPromptTemplate.from_messages([
        ("system", """Eres un 'Orchestrator' experto en un sistema de agentes de IA para un consultorio dental.
Tu función es enrutar la conversación al agente correcto basándote en la intención del usuario.
Las opciones de agentes son:
- Knowledge Concierge: Para consultas generales sobre precios, servicios, horarios, etc.
- Scheduler: Para agendar, reprogramar o cancelar una cita.
- end: Si la intención es 'invalido' o no requiere acción."""),
        ("human", "La intención del usuario es: '{intention}'. ¿A qué agente debo enrutar la conversación?"),
    ])

    chain = router_prompt | structured_llm
    route = chain.invoke({"intention": state["intention"]})
    print(f"Decisión del Router: dirigir a -> {route.next_node}")
    return {"next_node": route.next_node, "intention": state["intention"]}
