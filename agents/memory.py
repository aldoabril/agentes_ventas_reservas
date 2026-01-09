
from typing import Any, List
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage

# Número de mensajes a mantener en short-term memory
SHORT_HISTORY_LIMIT = 5


def push_message(
    messages: list[Any] | None, message: Any, limit: int = SHORT_HISTORY_LIMIT
) -> list[Any]:
    """Añade `message` a la lista `messages` y devuelve la lista truncada a `limit`.

    Args:
        messages: lista existente (puede ser None)
        message: nuevo mensaje a añadir
        limit: tamaño máximo de la ventana

    Returns:
        Lista de mensajes con como máximo `limit` elementos (últimos items).
    """
    if messages is None:
        messages = []

    messages.append(message)
    # mantener solo las últimas `limit` entradas
    if limit is not None and limit > 0:
        return messages[-limit:]
    return messages


def truncate_messages(
    messages: list[Any] | None, limit: int = SHORT_HISTORY_LIMIT
) -> list[Any] | None:
    """Devuelve una vista truncada (últimos `limit`) de `messages` o None si no hay mensajes."""
    if messages is None:
        return None
    if limit is None or limit <= 0:
        return messages
    return messages[-limit:]


def get_safe_history_window(messages: List[BaseMessage], max_messages: int = 10) -> List[BaseMessage]:
    """
    Obtiene una ventana segura de mensajes recientes para enviar al LLM.
    Asegura que la secuencia no comience con un mensaje inválido para modelos como Gemini.
    
    Reglas:
    1. Intentar obtener hasta `max_messages`.
    2. Si el corte cae en medio de una secuencia tool-call, retroceder o avanzar para corregir.
    3. Gemini NO permite empezar con AIMessage(tool_calls) si no hay un User antes.
    4. Gemini NO permite empezar con ToolMessage (huérfano).
    
    Estrategia simplificada:
    - Tomar `max_messages` del final.
    - Recortar desde el inicio de ese slice hasta encontrar un HumanMessage.
    - Si no hay HumanMessage en el slice, devolver todo el slice (mejor esfuerzo) o vaciarlo.
    """
    if not messages:
        return []
        
    # Paso 1: Slice inicial
    # Tomamos un poco más de margen para intentar encontrar un HumanMessage
    slice_size = max_messages + 5 
    recent = messages[-slice_size:]
    
    # Paso 2: Buscar el PRIMER HumanMessage dentro de los últimos `max_messages` (aprox)
    # Queremos que la lista resultante tenga como MUCHO `max_messages`, pero
    # priorizamos la validez sobre la longitud exacta.
    
    # Recorremos de atrás hacia adelante para encontrar un punto de corte válido.
    # Un punto válido es JUSTO ANTES de un HumanMessage.
    
    valid_start_index = -1
    count = 0
    
    for i in range(len(recent) - 1, -1, -1):
        msg = recent[i]
        count += 1
        
        if isinstance(msg, HumanMessage):
            # Encontramos un usuario. Este es un buen candidato para inicio del historial.
            valid_start_index = i
            
            # Si ya tenemos suficientes mensajes, podemos parar aquí.
            if count >= max_messages:
                break
    
    if valid_start_index != -1:
        return recent[valid_start_index:]
    
    # Fallback: Si no hay HumanMessage (raro en este flujo),
    # devolvemos el slice crudo normal, pero limpiamos inicios inválidos obvios.
    final_slice = messages[-max_messages:]
    
    # Limpieza básica de inicio
    while final_slice:
        first = final_slice[0]
        # Si empieza con ToolMessage -> Inválido, borrar.
        if isinstance(first, ToolMessage):
            final_slice.pop(0)
            continue
        # Si empieza con AI con ToolCalls -> Inválido para Gemini, borrar.
        if isinstance(first, AIMessage) and first.tool_calls:
            final_slice.pop(0)
            continue
        break
        
    return final_slice
