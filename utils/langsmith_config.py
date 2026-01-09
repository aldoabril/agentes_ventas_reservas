"""
Configuración y helpers para integración con LangSmith.
Proporciona callbacks, métricas y trazas automáticas.
"""
import os
from typing import Optional, Dict, Any, List
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.tracers import LangChainTracer
from langchain_core.tracers.context import tracing_v2_enabled
from langsmith import Client, traceable
from langsmith.run_helpers import tracing_context

from utils.logger import get_logger
from config.settings import LLM_PROVIDER

logger = get_logger(__name__)

# Cliente LangSmith global
_langsmith_client: Optional[Client] = None


def setup_langsmith() -> bool:
    """
    Configura LangSmith para trazas automáticas.
    
    Returns:
        True si LangSmith está configurado correctamente, False en caso contrario
    """
    global _langsmith_client
    
    # Verificar variables de entorno
    api_key = os.getenv("LANGCHAIN_API_KEY") or os.getenv("LANGSMITH_API_KEY")
    project = os.getenv("LANGCHAIN_PROJECT")
    endpoint = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
    tracing_enabled = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
    
    if not api_key:
        logger.warning("LANGCHAIN_API_KEY no configurada. LangSmith tracing deshabilitado.")
        return False
    
    if not tracing_enabled:
        logger.info("LANGCHAIN_TRACING_V2 no está en 'true'. LangSmith tracing deshabilitado.")
        return False
    
    try:
        # Configurar variables de entorno para LangChain
        os.environ["LANGCHAIN_API_KEY"] = api_key
        os.environ["LANGCHAIN_PROJECT"] = project
        os.environ["LANGCHAIN_ENDPOINT"] = endpoint
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        
        # Crear cliente LangSmith
        _langsmith_client = Client(api_key=api_key, api_url=endpoint)
        
        logger.info(f"LangSmith configurado correctamente. Proyecto: {project}")
        return True
    except Exception as e:
        logger.error(f"Error configurando LangSmith: {e}", exc_info=True)
        return False


def get_langsmith_client() -> Optional[Client]:
    """
    Obtiene el cliente LangSmith configurado.
    
    Returns:
        Cliente LangSmith o None si no está configurado
    """
    return _langsmith_client


def get_langsmith_callbacks(
    run_name: Optional[str] = None,
    tags: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> List[BaseCallbackHandler]:
    """
    Obtiene callbacks de LangSmith para usar en cadenas LangChain.
    
    Args:
        run_name: Nombre del run para identificar en LangSmith
        tags: Tags para categorizar el run
        metadata: Metadata adicional para el run
        
    Returns:
        Lista de callbacks de LangSmith
    """
    if not _langsmith_client:
        logger.debug("LangSmith no configurado, retornando callbacks vacíos")
        return []
    
    callbacks = []
    
    try:
        tracer = LangChainTracer(
            project_name=os.getenv("LANGCHAIN_PROJECT"),
            client=_langsmith_client
        )
        callbacks.append(tracer)
    except Exception as e:
        logger.warning(f"Error creando LangChainTracer: {e}")
    
    return callbacks


class LangSmithMetricsHandler(BaseCallbackHandler):
    """Callback handler personalizado para capturar métricas en LangSmith."""
    
    def __init__(self, agent_name: str, node_name: str):
        super().__init__()
        self.agent_name = agent_name
        self.node_name = node_name
        self.start_time: Optional[float] = None
        self.token_usage: Dict[str, int] = {}
    
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        """Registra inicio de llamada LLM."""
        import time
        self.start_time = time.time()
        logger.debug(f"LLM iniciado en {self.agent_name}/{self.node_name}")
    
    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        """Registra fin de llamada LLM y métricas."""
        import time
        if self.start_time:
            latency = time.time() - self.start_time
            logger.info(f"LLM completado en {latency:.2f}s en {self.agent_name}/{self.node_name}")
            
            # Capturar uso de tokens si está disponible
            if hasattr(response, "llm_output") and response.llm_output:
                token_info = response.llm_output.get("token_usage", {})
                if token_info:
                    self.token_usage = token_info
                    logger.debug(f"Tokens usados: {token_info}")
    
    def on_llm_error(self, error: Exception, **kwargs: Any) -> None:
        """Registra errores de LLM."""
        logger.error(f"Error en LLM {self.agent_name}/{self.node_name}: {error}", exc_info=True)


def trace_agent_execution(agent_name: str, node_name: str):
    """
    Decorador para trazar ejecución de agentes en LangSmith.
    
    Args:
        agent_name: Nombre del agente
        node_name: Nombre del nodo
        
    Returns:
        Decorador
    """
    def decorator(func):
        @traceable(name=f"{agent_name}.{node_name}", tags=[agent_name, node_name])
        def wrapper(*args, **kwargs):
            with tracing_context(
                tags=[agent_name, node_name, LLM_PROVIDER],
                metadata={"agent_name": agent_name, "node_name": node_name}
            ):
                return func(*args, **kwargs)
        return wrapper
    return decorator

