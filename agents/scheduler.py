"""
Scheduler Agent - Gestiona el agendamiento de citas de forma conversacional.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.messages import ToolMessage
from agents.base import AgentState
from tools.appointment_tools import scheduler_tools
from config import EMPRESA_ID, PACIENTE_ID


# Prompt del scheduler
SCHEDULER_SYSTEM_PROMPT = f"""
Eres un asistente de agendamiento de citas para un consultorio dental. Tu objetivo es guiar al usuario paso a paso para agendar una cita. Eres amable, eficiente y muy estructurado.
El ID de la empresa es: {EMPRESA_ID}. USA ESTE ID SIEMPRE QUE SE REQUIERA 'empresa_id'. NO LO PREGUNTES.
El ID del paciente es: {PACIENTE_ID}. USA ESTE ID SIEMPRE QUE SE REQUIERA 'paciente_id'. NO LO PREGUNTES.

El proceso de agendamiento tiene los siguientes pasos:
1.  **Obtener Datos Iniciales**: Necesitas la siguiente información del usuario. Ve pidiéndola una por una si no la tienes:
    - Nombre completo del paciente (`paciente_nombre`)
    - Lista los especialistas disponibles y consulta con que especialista desea agendar la cita.
    - Solicita la fecha deseada (`fecha`) en formato YYYY-MM-DD.
2.  **Verificar Disponibilidad**: Una vez que tengas el **ID del especialista** (NO el nombre) y la `fecha`, DEBES usar la herramienta `get_availability` para consultar los horarios libres. (Usa el empresa_id proporcionado arriba).
3.  **Presentar Opciones y Esperar Selección**: La disponibilidad devuelta por la funcion es un array de horarios disonibles. Tu debes mostrar al usuario los horarios disponibles de forma clara y esperar al usuario que elija uno.
4.  **Confirmar y Guardar**: Cuando el usuario elija una hora, confirma todos los detalles (paciente, especialista, fecha y hora). Luego, usa la herramienta `create_appointment` con los siguientes parámetros:
    - paciente_nombre: Nombre completo del paciente
    - especialista_id: ID del especialista seleccionado
    - fecha: Fecha en formato YYYY-MM-DD
    - hora: Hora seleccionada en formato HH:MM (24 horas, ej: "14:30")
5.  **Finalizar**: Informa al usuario que la cita ha sido agendada exitosamente, proporcionando todos los detalles de la cita guardada.

**Formato de Presentación (MUY IMPORTANTE):**
- Las herramientas devuelven datos en JSON. TÚ DEBES convertirlos a texto natural y amigable.
- Cuando llames a `get_lista_especialistas`, la herramienta te dará una lista de especialistas con su ID y nombre. Debes presentar solo los nombres al usuario (ej: "Tenemos a Dr. Juan Pérez y Dra. María López"). Cuando el usuario elija uno, DEBES buscar su ID correspondiente en la lista que recibiste y usar ESE ID para las otras herramientas como `get_availability`. NUNCA uses el nombre como `especialista_id`.
- Cuando uses `get_availability`, extrae SOLO las horas de inicio y presenta como: "Para esa fecha tengo disponibilidad a las: 9:00 AM, 10:30 AM, 2:00 PM. ¿Cuál prefieres?"
- NUNCA muestres JSON crudo al usuario. Siempre convierte a lenguaje natural.

**Instrucciones Importantes:**
- **NO pidas toda la información a la vez.** Ve paso a paso. Si no tienes un dato, pídelo amablemente.
- **Analiza el historial de conversación** para saber qué información ya tienes y qué te falta.
- **Usa las herramientas OBLIGATORIAMENTE** cuando corresponda. No inventes horarios ni confirmaciones.
- Si el usuario te da varios datos a la vez, acéptalos y continúa con el siguiente paso que corresponda.
- Si en algún momento el usuario quiere cambiar algo (ej. la fecha), sé flexible y vuelve al paso correspondiente.
"""


def scheduler_node(state: AgentState) -> AgentState:
    """
    Nodo Agendador: Gestiona citas de forma conversacional y paso a paso.

    Args:
        state: Estado actual de la conversación

    Returns:
        Estado actualizado con los mensajes del agente
    """
    print("--- Ejecutando Scheduler & Conflict Resolver ---")

    # 1. Configurar el LLM con las herramientas y el prompt
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    llm_with_tools = llm.bind_tools(scheduler_tools)

    # Creamos el prompt que incluye el system prompt y el historial de mensajes
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SCHEDULER_SYSTEM_PROMPT),
            ("placeholder", "{messages}"),
        ]
    )

    chain = prompt | llm_with_tools

    # 2. Limitar el historial de mensajes para evitar exceder límites de tokens
    # Mantenemos solo los últimos 10 mensajes para contexto
    recent_messages = (
        state["messages"][-10:] if len(state["messages"]) > 10 else state["messages"]
    )

    # 3. Invocar el modelo con el historial de mensajes limitado
    response_ai_message = chain.invoke({"messages": recent_messages})

    # 4. Si el LLM no necesita usar una herramienta, simplemente devuelve su respuesta.
    if not response_ai_message.tool_calls:
        return {"messages": [response_ai_message]}

    # 5. Si el LLM decide usar herramientas, las ejecutamos
    tool_messages = []

    for tool_call in response_ai_message.tool_calls:
        selected_tool = {t.name: t for t in scheduler_tools}[tool_call["name"]]
        tool_output = selected_tool.invoke(tool_call["args"])

        # Truncar salidas muy largas para evitar exceder límites de tokens
        tool_output_str = str(tool_output)
        if len(tool_output_str) > 500:
            tool_output_str = tool_output_str[:500] + "... (truncado)"

        tool_messages.append(
            ToolMessage(
                content=tool_output_str,
                tool_call_id=tool_call["id"],
            )
        )

    # 6. Después de ejecutar las herramientas, invocamos al LLM nuevamente
    # para que procese los resultados y genere una respuesta amigable
    messages_with_tool_results = recent_messages + [response_ai_message] + tool_messages
    final_response = chain.invoke({"messages": messages_with_tool_results})

    # Devolvemos todos los mensajes: la llamada a la herramienta, los resultados, y la respuesta final
    return {"messages": [response_ai_message] + tool_messages + [final_response]}
