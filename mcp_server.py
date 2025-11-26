# mcp_server.py
import asyncio
import sys
from fastmcp import FastMCP
from integrations import calendar_api
from typing import Dict, Any, Optional

# 1. Crea una instancia de FastMCP
# El 'name' es importante para la identificación del servidor.
mcp = FastMCP(
    name="CitasMedicasServer",
    instructions="Provee herramientas para consultar disponibilidad, agendar, modificar y cancelar citas médicas."
)

especialistas = {
    "GMcKghlgHvTkoPxj9t4X": "Dr. Juan Pérez",
    "oxFsA3phVDEVuw3jCQFx": "Dr. María López"
}
@mcp.tool()
def get_empresa_id(
) -> str:
    """
    Obtiene el ID de la empresa.
    """
    return "hIntsAEzBwy8Hwi4DNcf"


@mcp.tool()
def get_lista_especialistas(
) -> Dict[str, str]:
    """
    Obtiene la lista de especialistas.
    """
    return especialistas

@mcp.tool()
def get_especialista_nombre(
    especialista_id: str,
) -> str:
    """
    Obtiene el nombre de un especialista por su ID.
    """
    return especialistas.get(especialista_id, "Especialista no encontrado")

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
    return calendar_api.get_availability(empresa_id, especialista_id, fecha, paciente_id)

@mcp.tool()
def save_appointment(appointment_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Crea y agenda una nueva cita médica en el sistema.
    Se debe usar después de confirmar la disponibilidad y tener todos los datos del paciente.
    """
    return calendar_api.save_appointment(appointment_data)

@mcp.tool()
def update_appointment(
    appointment_id: str, appointment_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Modifica una cita médica ya existente. Sirve para reprogramar o cambiar detalles de una cita.
    """
    return calendar_api.update_appointment(appointment_id, appointment_data)

@mcp.tool()
def delete_appointment(appointment_id: str) -> Dict[str, Any]:
    """
    Elimina o cancela una cita médica existente del sistema usando su ID.
    """
    return calendar_api.delete_appointment(appointment_id)

@mcp.tool()
def find_appointments_by_patient(
    patient_id: str,
    fecha: Optional[str] = None,
) -> list[Dict[str, Any]]:
    """
    Busca y devuelve una lista de todas las citas agendadas para un paciente específico.
    Se puede filtrar por fecha.
    """
    return calendar_api.find_appointments_by_patient(patient_id, fecha)

@mcp.tool()
def get_specialist_schedule_config(specialist_id: str) -> Dict[str, Any]:
    """
    Obtiene la configuración general del horario de un especialista (días que trabaja, horas de inicio y fin).
    No devuelve la disponibilidad de un día, sino el horario base.
    """
    return calendar_api.get_specialist_schedule_config(specialist_id)


# --- Bloque para ejecutar el servidor ---
if __name__ == "__main__":
    try:
        # El servidor ahora se ejecutará sobre stdin/stdout, por lo que no se necesita host/puerto.
        # Asumimos que mcp.run es una corutina y necesita ser ejecutada en un bucle de eventos.
        # asyncio.run(mcp.run(transport="stdio"))
        mcp.run(transport="http", port=8000)
        print("Servidor finalizado.")
    except KeyboardInterrupt:
        print("\nServidor detenido.", file=sys.stderr)
    except Exception as e:
        print(f"Error al iniciar el servidor: {e}", file=sys.stderr)
