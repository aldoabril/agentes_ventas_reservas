"""
Herramientas para el agendamiento de citas.
Proporciona funciones para interactuar con el servidor MCP y gestionar citas.
"""
import asyncio
from typing import Dict, Any, Optional
from langchain_core.tools import tool
from fastmcp import Client

from config import EMPRESA_ID, PACIENTE_ID, MCP_SERVER_URL


async def call_mcp_tool(tool_name: str, arguments: Dict[str, Any]) -> Any:
    """
    Función auxiliar para conectarse al servidor MCP y llamar a una herramienta.
    
    Args:
        tool_name: Nombre de la herramienta MCP a llamar
        arguments: Argumentos para la herramienta
        
    Returns:
        Resultado de la herramienta o diccionario con error
    """
    print(f"--- Llamando a la herramienta MCP: {tool_name} con args: {arguments} ---")
    # Filtramos los argumentos que son None para no enviarlos
    arguments = {k: v for k, v in arguments.items() if v is not None}
    try:
        async with Client(MCP_SERVER_URL) as client:
            result = await client.call_tool(tool_name, arguments)
            if getattr(result, "isError", False):
                print(f"Error en la herramienta MCP '{tool_name}': {result.content}")
                return {"error": result.content}

            content = getattr(result, "structuredContent", None) or getattr(result, "content", None)
            if content is None:
                return {"status": "éxito", "respuesta": "La operación se completó sin un contenido de respuesta específico."}
            # Si el contenido es una lista de modelos Pydantic, los convertimos a dict
            if isinstance(content, list):
                return [item.model_dump() if hasattr(item, "model_dump") else item for item in content]

            return content

    except Exception as e:
        print(f"Error de conexión o ejecución en MCP: {e}")
        return {"error": f"No se pudo conectar o ejecutar la herramienta: {str(e)}"}


@tool
def get_lista_especialistas() -> Dict[str, Any]:
    """
    Obtiene la lista de especialistas disponibles.
    
    Returns:
        Diccionario con los IDs y nombres de los especialistas
    """
    return asyncio.run(call_mcp_tool("get_lista_especialistas", {}))


@tool
def get_especialista_nombre(especialista_id: str) -> str:
    """
    Obtiene el nombre de un especialista por su ID.
    
    Args:
        especialista_id: ID del especialista
        
    Returns:
        Nombre del especialista
    """
    return asyncio.run(call_mcp_tool("get_especialista_nombre", {"especialista_id": especialista_id}))


@tool
def get_availability(
    empresa_id: str,
    especialista_id: str,
    fecha: str,
    paciente_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Consulta y devuelve los horarios disponibles para un especialista en una fecha específica.
    Útil para saber qué horas se pueden agendar.
    Las citas vienen en formato JSON, muestra al usuario en listado de horas disponibles para la fecha seleccionada.
    
    Args:
        empresa_id: ID de la empresa
        especialista_id: ID del especialista
        fecha: Fecha en formato YYYY-MM-DD
        paciente_id: ID del paciente (opcional)
        
    Returns:
        Diccionario con los horarios disponibles
    """
    return asyncio.run(call_mcp_tool("get_availability", locals()))


@tool
def create_appointment(
    paciente_nombre: str,
    especialista_id: str,
    fecha: str,  # YYYY-MM-DD
    hora: str,   # HH:MM
    paciente_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Crea y agenda una nueva cita médica en el sistema.
    
    Args:
        paciente_nombre: Nombre completo del paciente
        especialista_id: ID del especialista seleccionado
        fecha: Fecha de la cita en formato YYYY-MM-DD
        hora: Hora de la cita en formato HH:MM (24h)
        paciente_id: ID del paciente si ya existe en el sistema (opcional)
    
    Ejemplo:
        create_appointment(
            paciente_nombre="Juan Pérez",
            especialista_id="GMcKghlgHvTkoPxj9t4X",
            fecha="2026-01-15",
            hora="10:30"
        )
        
    Returns:
        Diccionario con la información de la cita creada
    """
    # Importar la función helper desde integrations
    from integrations.calendar_api import build_appointment_data
    
    # Construir la estructura de la cita
    appointment_data = build_appointment_data(
        empresa_id=EMPRESA_ID,  # Usar el ID global desde config
        especialista_id=especialista_id,
        paciente_nombre=paciente_nombre,
        paciente_id=paciente_id,
        fecha=fecha,
        hora=hora,
    )
    
    # Llamar al MCP tool para guardar
    return asyncio.run(call_mcp_tool("save_appointment", {"appointment_data": appointment_data}))


# Lista de herramientas para el scheduler
scheduler_tools = [
    get_availability,
    get_lista_especialistas,
    get_especialista_nombre,
    create_appointment,
]
