"""
Decoradores para trazas automáticas y logging.
"""
import time
from functools import wraps
from typing import Callable, TypeVar, Any

from utils.logger import get_logger
from utils.langsmith_config import trace_agent_execution

T = TypeVar("T")

logger = get_logger(__name__)


def trace_llm_call(agent_name: str, node_name: str):
    """
    Decorador para trazar automáticamente llamadas a LLM.
    Captura inputs, outputs, latencia y errores.
    
    Args:
        agent_name: Nombre del agente
        node_name: Nombre del nodo
        
    Returns:
        Decorador
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        @trace_agent_execution(agent_name, node_name)
        def wrapper(*args, **kwargs) -> T:
            start_time = time.time()
            node_logger = get_logger(func.__module__, node_name=node_name)
            
            try:
                # Log de inicio
                node_logger.info(f"Iniciando llamada LLM en {agent_name}/{node_name}")
                
                # Ejecutar función
                result = func(*args, **kwargs)
                
                # Calcular latencia
                latency = time.time() - start_time
                
                # Log de éxito
                node_logger.info(
                    f"LLM completado exitosamente en {latency:.2f}s",
                    extra={"latency": latency, "agent": agent_name, "node": node_name}
                )
                
                return result
            except Exception as e:
                latency = time.time() - start_time
                node_logger.error(
                    f"Error en LLM después de {latency:.2f}s: {e}",
                    extra={"latency": latency, "agent": agent_name, "node": node_name},
                    exc_info=True
                )
                raise
        
        return wrapper
    return decorator


def log_execution_time(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorador simple para registrar tiempo de ejecución.
    
    Returns:
        Decorador
    """
    @wraps(func)
    def wrapper(*args, **kwargs) -> T:
        start_time = time.time()
        func_logger = get_logger(func.__module__)
        
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            func_logger.debug(f"{func.__name__} ejecutado en {execution_time:.2f}s")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            func_logger.error(f"{func.__name__} falló después de {execution_time:.2f}s: {e}", exc_info=True)
            raise
    
    return wrapper

