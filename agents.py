from typing import Literal, TypedDict, Annotated, Sequence
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage, AIMessage
from langgraph.graph.message import add_messages

# --- 1. Definición del Estado del Grafo (AgentState) ---
# El estado es un diccionario que se pasa entre los nodos del grafo.
# Contiene toda la información relevante de la conversación.

class AgentState(TypedDict):
    """
    Representa el estado de la conversación.

    Attributes:
        messages: La secuencia de mensajes que componen la conversación.
        intention: La intención clasificada por el Lead Qualifier.
        next_node: El siguiente nodo a ejecutar, determinado por el router.
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]
    intention: str
    next_node: str


# --- 2. Implementación de Nodos con Respuestas Dummy ---
# Cada nodo es una función que recibe el estado y devuelve una actualización del mismo.

class Intent(BaseModel):
    """Define el esquema para la clasificación de la intención."""
    intention: Literal["consulta", "reserva", "reprogramacion", "cancelacion", "objecion", "invalido"] = Field(
        description="La intención principal del mensaje del usuario."
    )

def lead_qualifier_node(state: AgentState) -> AgentState:
    """
    Nodo inicial: Clasifica la intención del usuario.
    Utiliza un LLM para determinar la intención del mensaje inicial.
    """
    print("--- Ejecutando Lead Qualifier ---")

    # Usamos un modelo eficiente para la clasificación
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    structured_llm = llm.with_structured_output(Intent)

    # Prompt para clasificar la intención
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

    chain = qualifier_prompt | structured_llm
    # Analizamos el último mensaje del usuario
    result = chain.invoke({"message": state["messages"][-1].content})

    state['intention'] = result.intention
    print(f"Intención detectada: {state['intention']}")
    return state

def knowledge_concierge_node(state: AgentState) -> AgentState:
    """
    Nodo de Conocimiento: Responde a consultas generales.
    Respuesta dummy.
    """
    print("--- Ejecutando Knowledge Concierge ---")
    response = "Hola, soy el Knowledge Concierge. Nuestros precios son competitivos. ¿En qué más puedo ayudarte?"
    # Devolvemos la respuesta como un mensaje de IA para añadirlo al historial
    return {"messages": [AIMessage(content=response)]}

def negotiator_node(state: AgentState) -> AgentState:
    """
    Nodo Negociador: Maneja objeciones.
    Respuesta dummy.
    """
    print("--- Ejecutando Negotiator & Objection Handler ---")
    response = "Entiendo tu objeción. Déjame ver qué alternativa puedo ofrecerte."
    return {"messages": [AIMessage(content=response)]}

def scheduler_node(state: AgentState) -> AgentState:
    """
    Nodo Agendador: Gestiona citas.
    Respuesta dummy.
    """
    print("--- Ejecutando Scheduler & Conflict Resolver ---")
    response = "Claro, estoy revisando la agenda para tu cita. Un momento por favor."
    return {"messages": [AIMessage(content=response)]}

def guardian_node(state: AgentState) -> AgentState:
    """
    Nodo Guardián: Valida acciones críticas.
    Respuesta dummy.
    """
    print("--- Ejecutando Guardian Agent ---")
    # En un caso real, este nodo no necesariamente respondería al usuario.
    # Podría modificar el estado o enrutar a otro nodo.
    # Por ahora, simulamos que no hace nada visible para el usuario.
    print("Acción validada internamente.")
    return {}

# --- 3. Lógica y Prompt del Router (Orchestrator) ---

class RouteQuery(BaseModel):
    """Define el esquema para la decisión de enrutamiento."""
    next_node: Literal["Knowledge Concierge", "Scheduler", "Negotiator", "Guardian", "end"] = Field(
        description="El nodo al que se debe dirigir la conversación a continuación."
    )

def router_node(state: AgentState) -> AgentState:
    """
    Nodo Router: Decide el siguiente paso basado en la intención.
    """
    print("--- Ejecutando Orchestrator (Router) ---")

    # Configuración del LLM para tomar la decisión de enrutamiento
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    structured_llm = llm.with_structured_output(RouteQuery)

    # Prompt para el LLM
    router_prompt = ChatPromptTemplate.from_messages([
        ("system", """Eres un 'Orchestrator' experto en un sistema de agentes de IA para un consultorio dental.
Tu función es enrutar la conversación al agente correcto basándote en la intención del usuario.

Las opciones de agentes son:
- Knowledge Concierge: Para consultas generales sobre precios, servicios, horarios, etc.
- Scheduler: Para agendar, reprogramar o cancelar una cita.
- Negotiator: Cuando el usuario presenta una objeción sobre el precio o las condiciones.
- Guardian: Si la solicitud es ambigua o requiere una validación especial.
- end: Si la intención es 'invalido' o no requiere acción."""),
        ("human", "La intención del usuario es: '{intention}'. ¿A qué agente debo enrutar la conversación?"),
    ])

    chain = router_prompt | structured_llm
    route = chain.invoke({"intention": state["intention"]})

    print(f"Decisión del Router: dirigir a -> {route.next_node}")
    state['next_node'] = route.next_node
    return state