"""
Orchestrator Agent - Router que decide el siguiente paso en la conversación.
"""

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from agents.base import AgentState, Intent, RouteQuery

def router_node(state: AgentState) -> dict:
    """
    Nodo Router: Decide el siguiente paso basado en la intención.
    Clasifica la intención (Context-Aware) si no existe y luego enruta al agente apropiado.
    """
    print("--- Ejecutando Orchestrator (Router) ---")

    # 1. Obtener mensajes y contexto
    messages = state["messages"]
    if not messages:
        return {"next_node": "end"}

    # Formatear historial para el LLM (Context Awareness)
    history_messages = messages[:-1][-5:] 
    history_text = "\n".join([f"{m.type}: {m.content}" for m in history_messages])
    
    last_message = messages[-1]
    user_input = last_message.content if isinstance(last_message, HumanMessage) else str(last_message)

    llm = ChatOpenAI(temperature=0, model="gpt-4o-mini") # Model rápido y capaz

    # 2. Clasificación de Intención (Siempre ejecutada ahora que Lead Qualifier no lo hace)
    #    NOTA: Si ya viniera pre-clasificada (ej. por algún otro mecanismo), podríamos saltar esto.
    #    Pero por diseño actual, Orchestrator es el ÚNICO clasificador de intención.
    
    print("--- Clasificando Intención (Context Aware) ---")
    
    intent_prompt_template = """Eres un experto clasificador de intenciones para una clínica dental.
Analiza el último mensaje del usuario Teniendo en cuenta el HISTORIAL.

CLASIFICACIONES:
- `consulta`: Info general, precios, servicios.
- `reserva`: Quiere una cita O está dando datos para una (nombre, fecha, etc).
- `reprogramacion`: Cambiar cita existente.
- `cancelacion`: Cancelar cita.
- `queja`: Insatisfacción con servicio/atención.
- `objecion`: Le parece caro, duda de condiciones.
- `invalido`: No se entiende nada (pero pasó filtro de seguridad).
- `otro`: Saludos, 'gracias', 'ok'.

IMPORTANTE:
- Si el usuario responde a una pregunta del asistente (ver historial), MANTÉN la intención del flujo (ej. si le preguntaron nombre, es parte de `reserva`).

Responde con JSON."""

    intent_prompt = ChatPromptTemplate.from_messages([
        ("system", intent_prompt_template),
        ("human", "Historial:\n{history}\n\nMensaje Actual: {input}"),
    ])

    intent_chain = intent_prompt | llm.with_structured_output(Intent)

    try:
        intent_result = intent_chain.invoke({"input": user_input, "history": history_text})
        current_intention = intent_result.intention
        print(f"Intención detectada: {current_intention}")
    except Exception as e:
        print(f"Error en Clasificación: {e}")
        current_intention = "otro"

    # Actualizar estado
    state["intention"] = current_intention

    # 3. Decision de Enrutamiento
    #    Configuración del LLM para tomar la decisión de enrutamiento
    structured_llm_router = llm.with_structured_output(RouteQuery)

    router_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """Eres un 'Orchestrator' experto en un sistema de agentes de IA para un consultorio dental.
Tu función es enrutar la conversación al agente correcto basándote en la intención del usuario.
Las opciones de agentes son:
- Knowledge Concierge: Para consultas generales sobre precios, servicios, horarios, etc. (Intención: consulta)
- Scheduler: Para agendar, reprogramar o cancelar una cita. (Intención: reserva, reprogramacion, cancelacion)
- end: Si la intención es 'invalido', 'otro', 'queja' (si no hay agente de soporte aun), o no requiere acción.
  - Para 'otro' (saludos), a veces el Scheduler puede responder si se quiere iniciar flujo, pero por seguridad 'end' o 'Knowledge Concierge' (para saludar) es mejor. 
  - Para este caso: Si es 'otro', envia a 'Knowledge Concierge' para que responda el saludo amablemente o de info general.
""",
            ),
            (
                "human",
                "La intención del usuario es: '{intention}'. ¿A qué agente debo enrutar la conversación?",
            ),
        ]
    )

    chain_router = router_prompt | structured_llm_router
    route = chain_router.invoke({"intention": current_intention})
    
    print(f"Decisión del Router: dirigir a -> {route.next_node}")
    return {"next_node": route.next_node, "intention": current_intention}
