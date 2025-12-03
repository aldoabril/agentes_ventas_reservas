"""
Definiciones base para el sistema de agentes.
Contiene tipos, estados y modelos compartidos.
"""

from typing import Literal, TypedDict, Annotated, Sequence, Optional
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class AgentState(TypedDict):
    """
    Representa el estado de la conversación, incluyendo los datos para agendar una cita.

    Attributes:
        messages: La secuencia de mensajes.
        intention: La intención clasificada del usuario.
        next_node: El siguiente nodo al que dirigir la conversación.
        empresa_id: ID de la empresa (opcional).
        paciente_nombre: Nombre del paciente (opcional).
        especialista_id: ID del especialista (opcional).
        fecha: Fecha deseada para la cita (opcional).
        horarios_disponibles: Lista de horarios disponibles (opcional).
        hora_seleccionada: Hora seleccionada para la cita (opcional).
        booking_complete: Flag para indicar si el proceso de reserva ha finalizado.
    """

    messages: Annotated[Sequence[BaseMessage], add_messages]
    intention: str
    next_node: str
    empresa_id: Optional[str]
    paciente_nombre: Optional[str]
    especialista_id: Optional[str]
    fecha: Optional[str]
    horarios_disponibles: Optional[list]
    hora_seleccionada: Optional[str]
    booking_complete: Optional[bool]


class Intent(BaseModel):
    """Define el esquema para la clasificación de la intención."""

    intention: Literal["consulta", "reserva"] = Field(
        description="La intención principal del mensaje del usuario."
    )


class RouteQuery(BaseModel):
    """Define el esquema para la decisión de enrutamiento."""

    next_node: Literal["Knowledge Concierge", "Scheduler", "end"] = Field(
        description="El nodo al que se debe dirigir la conversación a continuación."
    )
