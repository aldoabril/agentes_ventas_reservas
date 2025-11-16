# mcp_server.py

from typing import Any, Dict, List
from fastmcp import MCPServer, MCPTool, CallToolResult
import asyncio
import requests

# 1. Importa las funciones cliente que acabamos de crear
import api_clients

# 2. Mapea los nombres de las herramientas a las funciones reales
AVAILABLE_TOOLS = {
    "get_availability": api_clients.get_availability,
    "save_appointment": api_clients.save_appointment,
    "update_appointment": api_clients.update_appointment,
    "delete_appointment": api_clients.delete_appointment,
    "find_appointments_by_patient": api_clients.find_appointments_by_patient,
    "get_specialist_schedule_config": api_clients.get_specialist_schedule_config,
}

class AgentToolsMCPServer(MCPServer):
    """
    Un servidor MCP que expone las herramientas de gestión de citas médicas.
    """
    def name(self) -> str:
        return "CitasMedicasServer"

    async def list_tools(self) -> List[MCPTool]:
        """
        Describe las herramientas disponibles. La calidad de estas descripciones
        es CRUCIAL para que el agente decida correctamente.
        """
        tools = [
            MCPTool(
                name="get_availability",
                description="Consulta y devuelve los horarios disponibles para un especialista en una fecha específica. Útil para saber qué horas se pueden agendar.",
                arguments={
                    "empresa_id": "ID de la empresa o clínica.",
                    "especialista_id": "ID del especialista para el cual se consulta la disponibilidad.",
                    "fecha": "La fecha para la consulta en formato YYYY-MM-DD.",
                    "paciente_id": "(Opcional) ID del paciente, si es relevante para la disponibilidad."
                },
            ),
            MCPTool(
                name="save_appointment",
                description="Crea y agenda una nueva cita médica en el sistema. Se debe usar después de confirmar la disponibilidad y tener todos los datos del paciente.",
                arguments={
                    "appointment_data": "Un diccionario JSON con los detalles completos de la cita (ej: pacienteId, especialistaId, fecha, hora, motivo, etc.)."
                },
            ),
            MCPTool(
                name="update_appointment",
                description="Modifica una cita médica ya existente. Sirve para reprogramar o cambiar detalles de una cita.",
                arguments={
                    "appointment_id": "El ID de la cita que se desea modificar.",
                    "appointment_data": "Un diccionario JSON con los campos a actualizar."
                },
            ),
            MCPTool(
                name="delete_appointment",
                description="Elimina o cancela una cita médica existente del sistema usando su ID.",
                arguments={
                    "appointment_id": "El ID de la cita que se desea eliminar."
                },
            ),
            MCPTool(
                name="find_appointments_by_patient",
                description="Busca y devuelve una lista de todas las citas agendadas para un paciente específico. Se puede filtrar por fecha.",
                arguments={
                    "patient_id": "El ID del paciente cuyas citas se quieren encontrar.",
                    "fecha": "(Opcional) La fecha específica para filtrar las citas en formato YYYY-MM-DD."
                },
            ),
            MCPTool(
                name="get_specialist_schedule_config",
                description="Obtiene la configuración general del horario de un especialista (días que trabaja, horas de inicio y fin). No devuelve la disponibilidad de un día, sino el horario base.",
                arguments={
                    "specialist_id": "El ID del especialista cuyo horario base se quiere consultar."
                },
            ),
        ]
        return tools

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any] | None) -> CallToolResult:
        """
        Ejecuta la herramienta solicitada por el agente, delegando a la función cliente correspondiente.
        """
        if tool_name not in AVAILABLE_TOOLS:
            return CallToolResult(stdout=f"Error: Herramienta '{tool_name}' no encontrada.", stderr="Herramienta desconocida.")

        try:
            tool_function = AVAILABLE_TOOLS[tool_name]
            
            # Las funciones de `requests` no son asíncronas, por lo que no necesitamos `await`
            result = tool_function(**(arguments or {}))

            # Convertimos el resultado (que debería ser un dict o list) a un string para el stdout
            import json
            return CallToolResult(stdout=json.dumps(result, indent=2, ensure_ascii=False))

        except requests.exceptions.HTTPError as e:
            # Captura errores específicos de HTTP para dar una mejor respuesta
            error_body = e.response.text
            return CallToolResult(
                stdout=f"Error de API al ejecutar '{tool_name}': {e.response.status_code} {e.response.reason}. Detalles: {error_body}",
                stderr=str(e)
            )
        except Exception as e:
            return CallToolResult(
                stdout=f"Error inesperado al ejecutar la herramienta '{tool_name}': {e}",
                stderr=str(e)
            )

    # Métodos requeridos por la clase base abstracta
    async def connect(self): pass
    async def cleanup(self): pass

