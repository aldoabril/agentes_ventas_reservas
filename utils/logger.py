"""
Sistema de logging estructurado para el sistema de agentes.
Proporciona logging con formato JSON y contexto automático.
"""
import json
import logging
import os
from datetime import datetime
from typing import Optional, Dict, Any
from functools import wraps


class StructuredFormatter(logging.Formatter):
    """Formateador que genera logs en formato JSON estructurado."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Formatea el registro de log como JSON."""
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        # Añadir contexto adicional si está disponible
        if hasattr(record, "thread_id"):
            log_data["thread_id"] = record.thread_id
        if hasattr(record, "node_name"):
            log_data["node_name"] = record.node_name
        if hasattr(record, "agent_name"):
            log_data["agent_name"] = record.agent_name
        if hasattr(record, "extra_context"):
            log_data.update(record.extra_context)
        
        # Añadir información de excepción si existe
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_data, ensure_ascii=False)


class StandardFormatter(logging.Formatter):
    """Formateador estándar con información de contexto."""
    
    def __init__(self):
        super().__init__(
            fmt="%(asctime)s - %(name)s - %(levelname)s - [%(thread_id)s] [%(node_name)s] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
    
    def format(self, record: logging.LogRecord) -> str:
        """Formatea el registro añadiendo campos de contexto por defecto."""
        if not hasattr(record, "thread_id"):
            record.thread_id = "unknown"
        if not hasattr(record, "node_name"):
            record.node_name = "unknown"
        return super().format(record)


def setup_logging(
    log_level: Optional[str] = None,
    log_format: Optional[str] = None,
    log_file: Optional[str] = None
) -> None:
    """
    Configura el sistema de logging.
    
    Args:
        log_level: Nivel de logging (DEBUG, INFO, WARNING, ERROR). Por defecto desde env.
        log_format: Formato de log ('json' o 'standard'). Por defecto desde env.
        log_file: Archivo opcional para escribir logs. Si es None, solo stdout.
    """
    # Obtener configuración desde variables de entorno
    level_str = log_level or os.getenv("LOG_LEVEL", "INFO").upper()
    format_type = log_format or os.getenv("LOG_FORMAT", "standard").lower()
    
    # Convertir nivel de string a constante de logging
    level = getattr(logging, level_str, logging.INFO)
    
    # Configurar formateador
    if format_type == "json":
        formatter = StructuredFormatter()
    else:
        formatter = StandardFormatter()
    
    # Configurar handler para stdout
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)
    
    # Configurar logger raíz
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()
    root_logger.addHandler(console_handler)
    
    # Configurar handler de archivo si se especifica
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        root_logger.addHandler(file_handler)
    
    # Reducir verbosidad de librerías externas
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def get_logger(name: str, thread_id: Optional[str] = None, node_name: Optional[str] = None) -> logging.Logger:
    """
    Obtiene un logger con contexto pre-configurado.
    
    Args:
        name: Nombre del logger (típicamente __name__)
        thread_id: ID del thread de conversación
        node_name: Nombre del nodo del grafo
        
    Returns:
        Logger configurado
    """
    logger = logging.getLogger(name)
    
    # Añadir contexto al logger usando filtros
    class ContextFilter(logging.Filter):
        def __init__(self, thread_id: Optional[str], node_name: Optional[str]):
            super().__init__()
            self.thread_id = thread_id or "unknown"
            self.node_name = node_name or "unknown"
        
        def filter(self, record: logging.LogRecord) -> bool:
            record.thread_id = self.thread_id
            record.node_name = self.node_name
            return True
    
    # Añadir filtro si no existe
    if not any(isinstance(f, ContextFilter) for f in logger.filters):
        logger.addFilter(ContextFilter(thread_id, node_name))
    
    return logger


def log_function_call(logger: logging.Logger):
    """
    Decorador para registrar llamadas a funciones con contexto.
    
    Args:
        logger: Logger a usar para registrar
        
    Returns:
        Decorador
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger.debug(f"Llamando {func.__name__} con args={args}, kwargs={kwargs}")
            try:
                result = func(*args, **kwargs)
                logger.debug(f"{func.__name__} completado exitosamente")
                return result
            except Exception as e:
                logger.error(f"Error en {func.__name__}: {e}", exc_info=True)
                raise
        return wrapper
    return decorator

