"""
Scheduler Agent - Gestiona el agendamiento de citas de forma conversacional.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.messages import ToolMessage, AIMessage
from agents.base import AgentState
from tools.appointment_tools import scheduler_tools
from config import EMPRESA_ID, PACIENTE_ID
from config import GPT_MODELS, LLM_PROVIDER, GEMINI_MODELS
from langchain_google_genai import ChatGoogleGenerativeAI
from datetime import datetime

# Prompt del scheduler (TEMPLATE)
SCHEDULER_SYSTEM_PROMPT_TEMPLATE = """
Eres un asistente de agendamiento de citas para un consultorio dental. Tu objetivo es guiar al usuario paso a paso para agendar una cita. Eres amable, eficiente y muy estructurado.
La fecha actual es: {current_date} y la hora actual es: {current_time}.  No se puede agendar citas para fechas y horas anteriores a la fecha y hora actual.
El ID de la empresa es: {empresa_id}. USA ESTE ID SIEMPRE QUE SE REQUIERA 'empresa_id'. NO LO PREGUNTES.
El ID del paciente es: {paciente_id}. USA ESTE ID SIEMPRE QUE SE REQUIERA 'paciente_id'. NO LO PREGUNTES.


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
- Cuando llames a `get_lista_especialistas`, la herramienta te dará una lista de especialistas con su ID y nombre. Debes presentar solo los nombres al usuario (ej: "Tenemos a 1. Dr. Juan Pérez y 2. Dra. María López"). Cuando el usuario elija uno, DEBES buscar su ID correspondiente en la lista que recibiste y usar ESE ID para las otras herramientas como `get_availability`. NUNCA uses el nombre como `especialista_id`.
- Cuando uses `get_availability`, verifica primero que la fecha sea valida y que sea mayor igual a la fecha actual y hora actual. Si no es valida, informa al usuario que no se puede agendar para esa fecha y pide una nueva. Si es valida, extrae SOLO las horas de inicio y presenta como: "Para esa fecha tengo disponibilidad a las: 9:00 AM, 10:30 AM, 2:00 PM. ¿Cuál prefieres?"
- NUNCA muestres JSON crudo al usuario. Siempre convierte a lenguaje natural.

**Instrucciones Importantes:**
- **NO pidas toda la información a la vez.** Ve paso a paso. Si no tienes un dato, pídelo amablemente.
- **Analiza el historial de conversación** para saber qué información ya tienes y qué te falta.
- **Usa las herramientas OBLIGATORIAMENTE** cuando corresponda. No inventes horarios ni confirmaciones.
- Si el usuario te da varios datos a la vez, acéptalos y continúa con el siguiente paso que corresponda.
- Si el resultado de una herramienta indica un error (ej. fecha pasada), comunícalo al usuario y pide corregirlo.
- Si en algún momento el usuario quiere cambiar algo (ej. la fecha), sé flexible y vuelve al paso correspondiente.
"""


def scheduler_node(state: AgentState) -> dict:
    """
    Nodo Agendador: Gestiona citas de forma conversacional y paso a paso.
    Implementa un bucle 'ReAct' (Reason + Act) para iterar sobre llamadas a herramientas.

    Args:
        state: Estado actual de la conversación

    Returns:
        Estado actualizado con los mensajes del agente
    """
    print("--- Ejecutando Scheduler & Conflict Resolver (ReAct Loop) ---")

    # 1. Setup Básico
    now = datetime.now()
    current_date = now.strftime("%Y-%m-%d")
    current_time = now.strftime("%H:%M:%S")

    if LLM_PROVIDER == "gemini":
        llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODELS.GEMINI_25_FLASH.value,
            temperature=0,
            max_tokens=None,
            timeout=None,
        )
    else:
        llm = ChatOpenAI(temperature=0, model=GPT_MODELS.GPT_4O_MINI.value) 
    
    llm_with_tools = llm.bind_tools(scheduler_tools)

    formatted_system_prompt = SCHEDULER_SYSTEM_PROMPT_TEMPLATE.format(
        current_date=current_date,
        current_time=current_time,
        empresa_id=EMPRESA_ID,
        paciente_id=PACIENTE_ID
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", formatted_system_prompt),
        ("placeholder", "{messages}"),
    ])

    chain = prompt | llm_with_tools

    # Historial inicial
    messages = state["messages"][-10:] if len(state["messages"]) > 10 else state["messages"]
    
    # Recolector de mensajes GENERADOS en este turno (para devolver al grafo)
    new_messages = []

    # 2. Bucle ReAct
    MAX_ITERATIONS = 5
    iteration = 0
    
    while iteration < MAX_ITERATIONS:
        iteration += 1
        print(f"--- Iteración {iteration} ---")
        
        # Invocar LLM
        # Pasamos TODOS los mensajes acumulados hasta el momento (historial + lo que llevamos generado)
        current_messages_context = messages + new_messages
        response_ai = chain.invoke({"messages": current_messages_context})
        
        # Añadir respuesta del AI a la lista de mensajes nuevos
        new_messages.append(response_ai)
        
        # Si NO quiere usar herramientas -> Terminamos (es una respuesta al usuario)
        if not response_ai.tool_calls:
            print("--- Respuesta Final (sin tools) ---")
            break
            
        # Si QUIERE usar herramientas -> Ejecutarlas
        print(f"--- Ejecutando {len(response_ai.tool_calls)} herramientas ---")
        for tool_call in response_ai.tool_calls:
            try:
                # Buscar herramienta
                selected_tool = {t.name: t for t in scheduler_tools}[tool_call["name"]]
                
                # Ejecutar
                tool_output = selected_tool.invoke(tool_call["args"])
                
                # Truncar salida
                tool_output_str = str(tool_output)
                if len(tool_output_str) > 2000: # Aumenté un poco el límite
                    tool_output_str = tool_output_str[:2000] + "... (truncado)"
                
                print(f"Tool '{tool_call['name']}' Output: {tool_output_str[:100]}...")

            except Exception as e:
                tool_output_str = f"Error ejecutando herramienta: {str(e)}"
                print(f"Error Tool: {e}")

            # Crear ToolMessage
            tool_msg = ToolMessage(
                content=tool_output_str,
                tool_call_id=tool_call["id"],
                name=tool_call["name"]
            )
            
            # Añadir a la lista
            new_messages.append(tool_msg)
        
        # El ciclo continuará: El LLM verá el ToolMessage en la siguiente iteración ("Observe")

    # Devolver SOLO los mensajes nuevos añadidos en este turno
    return {"messages": new_messages}
