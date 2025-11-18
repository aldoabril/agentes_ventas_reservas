# mcp_server.py
import asyncio
import sys
from fastmcp import FastMCP
import api_clients
from typing import Dict, Any, Optional

# 1. Crea una instancia de FastMCP
# El 'name' es importante para la identificación del servidor.
mcp = FastMCP(
    name="CitasMedicasServer",
    instructions="Provee herramientas para consultar disponibilidad, agendar, modificar y cancelar citas médicas.",
)

# 2. Define las herramientas usando el decorador @mcp.tool()
# Las descripciones (docstrings) son cruciales para que el LLM sepa cómo usar la herramienta.

@mcp.tool()
def get_availability(
    empresa_id: str,
    especialista_id: str,
    fecha: str,
    paciente_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Consulta y devuelve los horarios disponibles para un especialista en una fecha específica.
    Útil para saber qué horas se pueden agendar.
    """
    return api_clients.get_availability(empresa_id, especialista_id, fecha, paciente_id)

@mcp.tool()
def save_appointment(appointment_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Crea y agenda una nueva cita médica en el sistema.
    Se debe usar después de confirmar la disponibilidad y tener todos los datos del paciente.
    """
    return api_clients.save_appointment(appointment_data)

@mcp.tool()
def update_appointment(
    appointment_id: str, appointment_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Modifica una cita médica ya existente. Sirve para reprogramar o cambiar detalles de una cita.
    """
    return api_clients.update_appointment(appointment_id, appointment_data)

@mcp.tool()
def delete_appointment(appointment_id: str) -> Dict[str, Any]:
    """
    Elimina o cancela una cita médica existente del sistema usando su ID.
    """
    return api_clients.delete_appointment(appointment_id)

@mcp.tool()
def find_appointments_by_patient(
    patient_id: str,
    fecha: Optional[str] = None,
) -> list[Dict[str, Any]]:
    """
    Busca y devuelve una lista de todas las citas agendadas para un paciente específico.
    Se puede filtrar por fecha.
    """
    return api_clients.find_appointments_by_patient(patient_id, fecha)

@mcp.tool()
def get_specialist_schedule_config(specialist_id: str) -> Dict[str, Any]:
    """
    Obtiene la configuración general del horario de un especialista (días que trabaja, horas de inicio y fin).
    No devuelve la disponibilidad de un día, sino el horario base.
    """
    return api_clients.get_specialist_schedule_config(specialist_id)


# --- Bloque para ejecutar el servidor ---
if __name__ == "__main__":
    try:
        # El servidor ahora se ejecutará sobre stdin/stdout, por lo que no se necesita host/puerto.
        # Asumimos que mcp.run es una corutina y necesita ser ejecutada en un bucle de eventos.
        # asyncio.run(mcp.run(transport="stdio"))
        mcp.run()
        print("Servidor finalizado.")
    except KeyboardInterrupt:
        print("\nServidor detenido.", file=sys.stderr)
    except Exception as e:
        print(f"Error al iniciar el servidor: {e}", file=sys.stderr)
