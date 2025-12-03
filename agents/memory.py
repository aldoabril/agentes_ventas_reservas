"""
Helpers para gestión de short-term memory (ventana deslizante) dentro del AgentState.

Estas utilidades mantienen la lista de mensajes limitada a una ventana (por defecto 5)
y proporcionan funciones seguras para manipular el estado conversacional desde
`main.py` o desde nodos de agentes.
"""

from typing import Any

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
