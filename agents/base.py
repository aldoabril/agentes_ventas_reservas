"""
Definiciones base para el sistema de agentes.
Contiene tipos, estados y modelos compartidos.
"""

from typing import Literal, TypedDict, Annotated, Sequence, Optional
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class AgentState(TypedDict, total=False):
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
        guardian_verdict: Verdicto del Guardian Agent (opcional).
        guardian_feedback: Feedback del Guardian para corrección (opcional).
        guardian_modifications: Modificaciones sugeridas por el Guardian (opcional).
        escalation_reason: Razón de escalación a humano (opcional).
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
    guardian_verdict: Optional[dict]
    guardian_feedback: Optional[str]
    guardian_modifications: Optional[str]
    escalation_reason: Optional[str]
    last_executed_node: Optional[str]  # Track which node last executed for Guardian
    guardian_rejection_count: Optional[int]  # Track number of rejections to prevent infinite loops
    rag_chunks: Optional[list]  # RAG documents retrieved by Knowledge Concierge
    rag_scores: Optional[list]   # Similarity scores for RAG documents


class Intent(BaseModel):
    """Define el esquema para la clasificación de la intención."""

    intention: Literal[
        "consulta",
        "reserva",
        "reprogramacion",
        "cancelacion",
        "queja",  # Lead Qualifier usa queja
        "objecion",  # Orchestrator usa objecion (podriamos unificar)
        "invalido",
        "otro",
    ] = Field(description="La intención principal del mensaje del usuario.")


class RouteQuery(BaseModel):
    """Define el esquema para la decisión de enrutamiento."""

    next_node: Literal["Knowledge Concierge", "Scheduler", "end"] = Field(
        description="El nodo al que se debe dirigir la conversación a continuación."
    )
