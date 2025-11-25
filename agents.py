from typing import Literal, TypedDict, Annotated, Sequence, Dict, Any, Optional
import asyncio

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage, AIMessage, ToolMessage
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from fastmcp import Client

# Importamos la cadena RAG desde el archivo retriever.py
from retriever import chain as rag_chain



# --- 1. Definición del Estado del Grafo (AgentState) ---

# El estado es un diccionario que se pasa entre los nodos del grafo.

# Contiene toda la información relevante de la conversación.



class AgentState(TypedDict):
    """
    Representa el estado de la conversación, incluyendo los datos para agendar una cita.
    Attributes:
        messages: La secuencia de mensajes.
        intention: La intención clasificada del usuario.
        next_node: El siguiente nodo a ejecutar.
        next_step_prompt: Instrucción para el siguiente paso que el agente debe solicitar.
        empresa_id: ID de la empresa (opcional).
        paciente_nombre: Nombre del paciente (opcional).
        especialista_id: ID del especialista (opcional).
        fecha: Fecha deseada para la cita (opcional).
        horarios_disponibles: Lista de horarios disponibles (opcional).
        hora_seleccionada: La hora que el usuario ha elegido (opcional).
        booking_complete: Flag para indicar si el proceso de reserva ha finalizado.
    """


    messages: Annotated[Sequence[BaseMessage], add_messages]
    intention: str
    next_node: str
    next_step_prompt: Optional[str]  # <-- Añadido
    empresa_id: Optional[str]
    paciente_nombre: Optional[str]
    especialista_id: Optional[str]
    fecha: Optional[str]
    horarios_disponibles: Optional[list]
    hora_seleccionada: Optional[str]
    booking_complete: bool


# --- 2. Definición de Herramientas para el Scheduler ---

async def call_mcp_tool(tool_name: str, arguments: Dict[str, Any]) -> Any:

    """Función auxiliar para conectarse al servidor MCP y llamar a una herramienta."""

    print(f"--- Llamando a la herramienta MCP: {tool_name} con args: {arguments} ---")

    # Filtramos los argumentos que son None para no enviarlos

    arguments = {k: v for k, v in arguments.items() if v is not None}

    try:

        async with Client("http://localhost:8000/mcp") as client:

            result = await client.call_tool(tool_name, arguments)

            

            if getattr(result, "isError", False):

                print(f"Error en la herramienta MCP '{tool_name}': {result.content}")

                return {"error": result.content}



            content = getattr(result, "structuredContent", None) or getattr(result, "content", None)

            

            if content is None:

                return {"status": "éxito", "respuesta": "La operación se completó sin un contenido de respuesta específico."}

            

            # Si el contenido es una lista de modelos Pydantic, los convertimos a dict

            if isinstance(content, list):

                return [item.model_dump() if hasattr(item, "model_dump") else item for item in content]

            

            return content



    except Exception as e:

        print(f"Error de conexión o ejecución en MCP: {e}")

        return {"error": f"No se pudo conectar o ejecutar la herramienta: {str(e)}"}



# Usamos el decorador @tool para que LangChain reconozca estas funciones como herramientas

@tool

def get_availability(

    empresa_id: str,

    especialista_id: str,

    fecha: str,

    paciente_id: Optional[str] = None,

) -> Dict[str, Any]:

    """

    Consulta y devuelve los horarios disponibles para un especialista en una fecha específica.

    Útil para saber qué horas se pueden agendar.

    """

    return asyncio.run(call_mcp_tool("get_availability", locals()))



@tool

def save_appointment(appointment_data: Dict[str, Any]) -> Dict[str, Any]:

    """

    Crea y agenda una nueva cita médica en el sistema.

    Se debe usar después de confirmar la disponibilidad y tener todos los datos del paciente.

    """

    return asyncio.run(call_mcp_tool("save_appointment", {"appointment_data": appointment_data}))



@tool

def update_appointment(

    appointment_id: str, appointment_data: Dict[str, Any]

) -> Dict[str, Any]:

    """

    Modifica una cita médica ya existente. Sirve para reprogramar o cambiar detalles de una cita.

    """

    return asyncio.run(call_mcp_tool("update_appointment", {"appointment_id": appointment_id, "appointment_data": appointment_data}))



@tool

def delete_appointment(appointment_id: str) -> Dict[str, Any]:

    """

    Elimina o cancela una cita médica existente del sistema usando su ID.

    """

    return asyncio.run(call_mcp_tool("delete_appointment", {"appointment_id": appointment_id}))



@tool

def find_appointments_by_patient(

    patient_id: str,

    fecha: Optional[str] = None,

) -> list[Dict[str, Any]]:

    """



    Busca y devuelve una lista de todas las citas agendadas para un paciente específico.

    Se puede filtrar por fecha.

    """

    return asyncio.run(call_mcp_tool("find_appointments_by_patient", locals()))



# Agrupamos las herramientas en una lista para el agente

scheduler_tools = [

    get_availability,

    save_appointment,

    update_appointment,

    delete_appointment,

    find_appointments_by_patient,

]





# --- 3. Implementación de Nodos del Grafo ---



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
    Nodo de Conocimiento: Responde a consultas generales utilizando RAG.
    """
    print("--- Ejecutando Knowledge Concierge ---")
    # Extraer la última pregunta del usuario del estado
    user_question = state["messages"][-1].content
    # Invocar la cadena RAG con la pregunta del usuario
    response = rag_chain.invoke(user_question)
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
    Nodo Agendador: Gestiona citas de forma conversacional y paso a paso.
    """
    print("--- Ejecutando Scheduler & Conflict Resolver ---")

   # Prompt mejorado para guiar al LLM en el proceso de agendamiento.
    SCHEDULER_SYSTEM_PROMPT = """
Eres un asistente de agendamiento de citas para un consultorio dental. Tu objetivo es guiar al usuario paso a paso para agendar una cita. Eres amable, eficiente y muy estructurado.
El proceso de agendamiento tiene los siguientes pasos:
1.  **Obtener Datos Iniciales**: Necesitas la siguiente información del usuario. Ve pidiéndola una por una si no la tienes:
    - Consulta con que especialista desea agendar la cita.
    - Obten el ID de la empresa de las herramientas disponibles.
    - Nombre completo del paciente (`paciente_nombre`)
    - Fecha deseada (`fecha`) en formato YYYY-MM-DD.
2.  **Verificar Disponibilidad**: Una vez que tengas `empresa_id`, `especialista_id` y `fecha`, DEBES usar la herramienta `get_availability` para consultar los horarios libres.
3.  **Presentar Opciones y Esperar Selección**: Muestra al usuario los horarios disponibles de forma clara y espera a que elija uno.
4. LAGU-CONFIRMACION**Confirmar y Guardar**: Cuando el usuario elija una hora, confirma todos los detalles (paciente, especialista, fecha y hora). Luego, usa la herramienta `save_appointment`. El argumento `appointment_data` para esta herramienta debe ser un diccionario que contenga: `empresa_id`, `especialista_id`, `paciente_nombre`, `fecha` y `hora`.
5.  **Finalizar**: Informa al usuario que la cita ha sido agendada exitosamente, proporcionando todos los detalles de la cita guardada.
**Instrucciones Importantes:**
- **NO pidas toda la información a la vez.** Ve paso a paso. Si no tienes un dato, pídelo amablemente.
- **Analiza el historial de conversación** para saber qué information ya tienes y qué te falta.
- **Usa las herramientas OBLIGATORIAMENTE** cuando corresponda. No inventes horarios ni confirmaciones.
- Si el usuario te da varios datos a la vez, acéptalos y continúa con el siguiente paso que corresponda.
- Si en algún momento el usuario quiere cambiar algo (ej. la fecha), sé flexible y vuelve al paso correspondiente.
"""

    # 1. Configurar el LLM con las herramientas y el nuevo prompt
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    llm_with_tools = llm.bind_tools(scheduler_tools)

    # Creamos el prompt que incluye el system prompt y el historial de mensajes
    prompt = ChatPromptTemplate.from_messages([
        ("system", SCHEDULER_SYSTEM_PROMPT),
        ("placeholder", "{messages}"),
    ])

    chain = prompt | llm_with_tools

    # 2. Invocar el modelo con el historial de mensajes actual
    response_ai_message = chain.invoke({"messages": state["messages"]})

    # 3. Si el LLM no necesita usar una herramienta, simplemente devuelve su respuesta.
    if not response_ai_message.tool_calls:
        return {"messages": [response_ai_message]}

    # 4. Si el LLM decide usar herramientas, las ejecutamos
    tool_messages = []

    for tool_call in response_ai_message.tool_calls:
        selected_tool = {t.name: t for t in scheduler_tools}[tool_call["name"]]
        tool_output = selected_tool.invoke(tool_call["args"])
        tool_messages.append(
            ToolMessage(
                content=str(tool_output),
                tool_call_id=tool_call["id"],
            )
        )

    # Devolvemos la llamada a la herramienta y su resultado para que el LLM
    # pueda continuar con el siguiente paso en la próxima invocación.

    return {"messages": [response_ai_message] + tool_messages}


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



# --- 4. Lógica y Prompt del Router (Orchestrator) ---

class RouteQuery(BaseModel):
    """Define el esquema para la decisión de enrutamiento."""
    next_node: Literal["Knowledge Concierge", "Scheduler", "Negotiator", "Guardian", "end"] = Field(
        description="El nodo al que se debe dirigir la conversación a continuación."

    )



def router_node(state: AgentState) -> dict:
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



    return {"next_node": route.next_node}
