"""
Sistema de retroalimentación para capturar feedback del usuario.
Integra con LangSmith para almacenar y analizar feedback.
"""
import os
from typing import Optional, Dict, Any, Literal
from datetime import datetime
from langsmith import Client, feedback

from utils.logger import get_logger
from utils.langsmith_config import get_langsmith_client

logger = get_logger(__name__)


class FeedbackCollector:
    """Recolector de feedback del usuario."""
    
    def __init__(self, thread_id: Optional[str] = None):
        """
        Args:
            thread_id: ID del thread de conversación
        """
        self.thread_id = thread_id
        self.client = get_langsmith_client()
        self.feedback_storage: list = []  # Almacenamiento local si LangSmith no está disponible
    
    def thumbs_up(self, run_id: Optional[str] = None, comment: Optional[str] = None) -> bool:
        """
        Registra feedback positivo (thumbs up).
        
        Args:
            run_id: ID del run en LangSmith (opcional)
            comment: Comentario adicional (opcional)
            
        Returns:
            True si se registró exitosamente
        """
        return self._record_feedback("positive", run_id, comment, score=1.0)
    
    def thumbs_down(self, run_id: Optional[str] = None, comment: Optional[str] = None) -> bool:
        """
        Registra feedback negativo (thumbs down).
        
        Args:
            run_id: ID del run en LangSmith (opcional)
            comment: Comentario adicional (opcional)
            
        Returns:
            True si se registró exitosamente
        """
        return self._record_feedback("negative", run_id, comment, score=0.0)
    
    def rating(self, score: float, run_id: Optional[str] = None, comment: Optional[str] = None) -> bool:
        """
        Registra un rating numérico.
        
        Args:
            score: Puntuación entre 0.0 y 1.0
            run_id: ID del run en LangSmith (opcional)
            comment: Comentario adicional (opcional)
            
        Returns:
            True si se registró exitosamente
        """
        # Normalizar score a rango 0-1
        normalized_score = max(0.0, min(1.0, score))
        return self._record_feedback("rating", run_id, comment, score=normalized_score)
    
    def comment(self, text: str, run_id: Optional[str] = None) -> bool:
        """
        Registra un comentario de texto.
        
        Args:
            text: Texto del comentario
            run_id: ID del run en LangSmith (opcional)
            
        Returns:
            True si se registró exitosamente
        """
        return self._record_feedback("comment", run_id, text, score=None)
    
    def _record_feedback(
        self,
        feedback_type: Literal["positive", "negative", "rating", "comment"],
        run_id: Optional[str],
        comment: Optional[str],
        score: Optional[float] = None
    ) -> bool:
        """
        Registra feedback en LangSmith o almacenamiento local.
        
        Args:
            feedback_type: Tipo de feedback
            run_id: ID del run
            comment: Comentario
            score: Puntuación (opcional)
            
        Returns:
            True si se registró exitosamente
        """
        feedback_data = {
            "type": feedback_type,
            "timestamp": datetime.utcnow().isoformat(),
            "thread_id": self.thread_id,
            "run_id": run_id,
            "comment": comment,
            "score": score,
        }
        
        # Intentar enviar a LangSmith si está disponible
        if self.client and run_id:
            try:
                # Crear feedback en LangSmith
                self.client.create_feedback(
                    run_id=run_id,
                    key=feedback_type,
                    score=score,
                    comment=comment,
                    value={
                        "thread_id": self.thread_id,
                        "timestamp": feedback_data["timestamp"],
                    }
                )
                logger.info(f"Feedback {feedback_type} registrado en LangSmith para run {run_id}")
                return True
            except Exception as e:
                logger.warning(f"Error enviando feedback a LangSmith: {e}. Almacenando localmente.")
        
        # Almacenar localmente como fallback
        self.feedback_storage.append(feedback_data)
        logger.info(f"Feedback {feedback_type} almacenado localmente")
        return True
    
    def get_feedback_summary(self) -> Dict[str, Any]:
        """
        Obtiene un resumen del feedback almacenado localmente.
        
        Returns:
            Diccionario con resumen de feedback
        """
        if not self.feedback_storage:
            return {"total": 0, "positive": 0, "negative": 0, "ratings": []}
        
        positive = sum(1 for f in self.feedback_storage if f["type"] == "positive")
        negative = sum(1 for f in self.feedback_storage if f["type"] == "negative")
        ratings = [f["score"] for f in self.feedback_storage if f["type"] == "rating" and f["score"] is not None]
        
        return {
            "total": len(self.feedback_storage),
            "positive": positive,
            "negative": negative,
            "ratings": ratings,
            "average_rating": sum(ratings) / len(ratings) if ratings else None,
        }


def create_feedback_collector(thread_id: Optional[str] = None) -> FeedbackCollector:
    """
    Crea una instancia de FeedbackCollector.
    
    Args:
        thread_id: ID del thread de conversación
        
    Returns:
        Instancia de FeedbackCollector
    """
    return FeedbackCollector(thread_id=thread_id)

