"""
Manejo robusto de errores para el sistema de agentes.
Incluye excepciones personalizadas, retry logic y circuit breaker.
"""
import time
import logging
from typing import Callable, TypeVar, Optional, Any, Dict
from functools import wraps
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    RetryError,
)

from utils.logger import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


# Excepciones personalizadas
class LLMError(Exception):
    """Error relacionado con llamadas a LLM."""
    def __init__(self, message: str, original_error: Optional[Exception] = None, context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.original_error = original_error
        self.context = context or {}


class ToolError(Exception):
    """Error relacionado con ejecución de herramientas."""
    def __init__(self, tool_name: str, message: str, original_error: Optional[Exception] = None):
        super().__init__(f"Error en herramienta '{tool_name}': {message}")
        self.tool_name = tool_name
        self.original_error = original_error


class ValidationError(Exception):
    """Error de validación de datos."""
    pass


# Circuit Breaker simple
class CircuitBreaker:
    """Circuit breaker simple para prevenir cascadas de errores."""
    
    def __init__(self, failure_threshold: int = 5, timeout: float = 60.0):
        """
        Args:
            failure_threshold: Número de fallos antes de abrir el circuito
            timeout: Tiempo en segundos antes de intentar cerrar el circuito
        """
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = "closed"  # closed, open, half_open
    
    def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Ejecuta una función con protección de circuit breaker.
        
        Args:
            func: Función a ejecutar
            *args, **kwargs: Argumentos para la función
            
        Returns:
            Resultado de la función
            
        Raises:
            Exception: Si el circuito está abierto o la función falla
        """
        # Verificar estado del circuito
        if self.state == "open":
            if self.last_failure_time and (time.time() - self.last_failure_time) > self.timeout:
                # Intentar cerrar el circuito (half-open)
                self.state = "half_open"
                logger.warning("Circuit breaker entrando en estado half-open")
            else:
                raise Exception(f"Circuit breaker está abierto. Reintentar después de {self.timeout}s")
        
        # Ejecutar función
        try:
            result = func(*args, **kwargs)
            # Éxito: resetear contador y cerrar circuito
            if self.state == "half_open":
                logger.info("Circuit breaker cerrado después de éxito en half-open")
            self.state = "closed"
            self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "open"
                logger.error(f"Circuit breaker abierto después de {self.failure_count} fallos")
            
            raise


# Circuit breakers globales por tipo de operación
_llm_circuit_breaker = CircuitBreaker(failure_threshold=5, timeout=60.0)
_tool_circuit_breaker = CircuitBreaker(failure_threshold=3, timeout=30.0)


def retry_with_backoff(
    max_attempts: int = 3,
    initial_wait: float = 1.0,
    max_wait: float = 10.0,
    exponential_base: float = 2.0,
    retry_on: tuple = (Exception,),
    use_circuit_breaker: bool = True
):
    """
    Decorador para retry con exponential backoff.
    
    Args:
        max_attempts: Número máximo de intentos
        initial_wait: Tiempo de espera inicial en segundos
        max_wait: Tiempo máximo de espera en segundos
        exponential_base: Base exponencial para el backoff
        retry_on: Tupla de excepciones para las que se debe reintentar
        use_circuit_breaker: Si usar circuit breaker
        
    Returns:
        Decorador
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            circuit_breaker = _llm_circuit_breaker if use_circuit_breaker else None
            
            @retry(
                stop=stop_after_attempt(max_attempts),
                wait=wait_exponential(multiplier=initial_wait, max=max_wait, exp_base=exponential_base),
                retry=retry_if_exception_type(retry_on),
                reraise=True
            )
            def _retry_func(*args, **kwargs):
                if circuit_breaker:
                    return circuit_breaker.call(func, *args, **kwargs)
                return func(*args, **kwargs)
            
            try:
                return _retry_func(*args, **kwargs)
            except RetryError as e:
                logger.error(f"Falló después de {max_attempts} intentos: {func.__name__}", exc_info=e)
                raise LLMError(
                    f"Error después de {max_attempts} intentos en {func.__name__}",
                    original_error=e.last_attempt.exception() if e.last_attempt else None
                )
            except Exception as e:
                logger.error(f"Error no recuperable en {func.__name__}: {e}", exc_info=True)
                raise
        
        return wrapper
    return decorator


def handle_llm_error(error: Exception, context: Optional[Dict[str, Any]] = None) -> str:
    """
    Maneja errores de LLM y retorna un mensaje de fallback apropiado.
    
    Args:
        error: Excepción capturada
        context: Contexto adicional del error
        
    Returns:
        Mensaje de error amigable para el usuario
    """
    logger.error(f"Error de LLM: {error}", extra={"context": context}, exc_info=True)
    
    # Mensajes de fallback según el tipo de error
    if isinstance(error, LLMError):
        if "timeout" in str(error).lower() or "timed out" in str(error).lower():
            return (
                "Lo siento, la solicitud está tomando más tiempo del esperado. "
                "Por favor, intenta nuevamente en un momento."
            )
        elif "rate limit" in str(error).lower() or "quota" in str(error).lower():
            return (
                "Estamos experimentando una alta demanda en este momento. "
                "Por favor, intenta nuevamente en unos segundos."
            )
        else:
            return (
                "Lo siento, tuve un problema al procesar tu consulta. "
                "¿Podrías intentar reformularla o contactarnos directamente?"
            )
    elif isinstance(error, ToolError):
        return (
            f"Hubo un problema al ejecutar una operación ({error.tool_name}). "
            "Por favor, intenta nuevamente o contacta con soporte."
        )
    else:
        return (
            "Lo siento, ocurrió un error inesperado. "
            "Nuestro equipo ha sido notificado. ¿Podrías intentar nuevamente?"
        )


def safe_execute(func: Callable[..., T], *args, fallback: Optional[T] = None, **kwargs) -> Optional[T]:
    """
    Ejecuta una función de forma segura, capturando excepciones.
    
    Args:
        func: Función a ejecutar
        *args: Argumentos posicionales
        fallback: Valor a retornar en caso de error
        **kwargs: Argumentos nombrados
        
    Returns:
        Resultado de la función o fallback
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logger.error(f"Error en ejecución segura de {func.__name__}: {e}", exc_info=True)
        return fallback

