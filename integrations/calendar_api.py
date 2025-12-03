# api_clients.py

import requests
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

# --- Configuración de la API ---
# Modifica esta URL para que apunte a la dirección base de tu API de citas.
# La he deducido del código del router de Express que proporcionaste.
API_BASE_URL = "https://us-central1-odontoplus-4db47.cloudfunctions.net/api/v1/citas"

# Zona horaria de la clínica (equivalente a clinicTz en TypeScript)
CLINIC_TIMEZONE = "America/Lima"


def build_appointment_data(
    empresa_id: str,
    especialista_id: str,
    paciente_nombre: str,
    paciente_id: Optional[str],
    fecha: str,  # YYYY-MM-DD
    hora: str,  # HH:MM formato 24h
    duracion_minutos: int = 60,
    summary: str = "Cita Dental",
    description: str = "Consulta general",
) -> Dict[str, Any]:
    """
    Construye la estructura de datos de una cita siguiendo el formato esperado por la API.

    Adaptado del código TypeScript mapperEventData().

    Args:
        empresa_id: ID de la empresa/clínica
        especialista_id: ID del especialista
        paciente_nombre: Nombre completo del paciente
        paciente_id: ID del paciente (opcional si es nuevo)
        fecha: Fecha en formato YYYY-MM-DD
        hora: Hora en formato HH:MM (24h)
        duracion_minutos: Duración de la cita en minutos (default: 60)
        summary: Título de la cita
        description: Descripción de la cita

    Returns:
        Dict con la estructura completa de la cita
    """
    # Construir datetime local de la clínica
    datetime_str = f"{fecha}T{hora}:00"
    clinic_tz = ZoneInfo(CLINIC_TIMEZONE)

    # Crear momento de inicio en zona horaria de la clínica
    start_local = datetime.fromisoformat(datetime_str).replace(tzinfo=clinic_tz)

    # Calcular momento de fin
    end_local = start_local + timedelta(minutes=duracion_minutos)

    # Convertir a UTC para enviar al backend
    start_utc = start_local.astimezone(ZoneInfo("UTC"))
    end_utc = end_local.astimezone(ZoneInfo("UTC"))

    # Construir estructura de la cita
    appointment_data = {
        "empresaId": empresa_id,
        "summary": summary,
        "description": description,
        # Fechas en UTC (como en TypeScript)
        "start": {
            "dateTime": start_utc.isoformat(),
            "timeZone": "UTC",
        },
        "end": {
            "dateTime": end_utc.isoformat(),
            "timeZone": "UTC",
        },
        # Asistentes
        "attendees": [
            {
                "email": f"paciente@temp.com",  # Placeholder, ajustar según necesidad
                "displayName": paciente_nombre,
            }
        ],
        # Metadatos
        "create_at": datetime.now(ZoneInfo("UTC")).isoformat(),
        "patientId": paciente_id,
        "especialistaId": especialista_id,
        "available": False,
        # Propiedades extendidas (para Google Calendar)
        "extendedProperties": {
            "private": {
                "businessId": empresa_id,
                "specialistId": especialista_id,
            }
        },
        # Opcional: representaciones locales para logging/debugging
        "startLocal": start_local.isoformat(),
        "endLocal": end_local.isoformat(),
    }

    return appointment_data


def get_availability(
    empresa_id: str,
    especialista_id: str,
    fecha: str,
    paciente_id: Optional[str] = None,
    auth_token: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Llama a la API para obtener la disponibilidad de un especialista.
    Corresponde a un día, la respuesta incluye los horarios disponibles para ese día."

    """
    params = {
        "empresaId": empresa_id,
        "especialistaId": especialista_id,
        "fechaIni": fecha,
        "fechaFin": fecha,
    }
    if paciente_id:
        params["pacienteId"] = paciente_id

    headers = {"Content-Type": "application/json"}
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"

    url = f"{API_BASE_URL}/dia"

    response = requests.get(url, params=params, headers=headers)

    # Convertir la respuesta a diccionario
    data = response.json()

    # Extraer solo las horas de inicio (dateTimeLocal) de citas disponibles
    horarios_disponibles = [
        cita["start"]["dateTimeLocal"]
        for cita in data.get("citas", [])
        if cita.get("available")
    ]

    print("Horarios disponibles:", horarios_disponibles)

    response.raise_for_status()  # Lanza una excepción para errores HTTP (4xx o 5xx)
    return {"horarios_disponibles": horarios_disponibles}


def save_appointment(appointment_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Llama a la API para crear una nueva cita.
    Corresponde a: POST /
    """
    response = requests.post(API_BASE_URL, json=appointment_data)
    response.raise_for_status()
    return response.json()


def update_appointment(
    appointment_id: str, appointment_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Llama a la API para actualizar una cita existente.
    Corresponde a: PUT /:id
    """
    url = f"{API_BASE_URL}/{appointment_id}"
    response = requests.put(url, json=appointment_data)
    response.raise_for_status()
    return response.json()


def delete_appointment(appointment_id: str) -> Dict[str, Any]:
    """
    Llama a la API para eliminar una cita.
    Corresponde a: DELETE /:id
    """
    url = f"{API_BASE_URL}/{appointment_id}"
    response = requests.delete(url)
    response.raise_for_status()
    return response.json()


def find_appointments_by_patient(
    patient_id: str, fecha: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Llama a la API para buscar citas por ID de paciente.
    Corresponde a: GET /:pacienteId
    """
    url = f"{API_BASE_URL}/{patient_id}"
    params = {}
    if fecha:
        params["fecha"] = fecha

    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()


def get_specialist_schedule_config(specialist_id: str) -> Dict[str, Any]:
    """
    Llama a la API para obtener la configuración de horario de un especialista.
    Corresponde a: GET /:id/configure

    ADVERTENCIA: Como se mencionó anteriormente, la ruta de la API de Express para esto
    probablemente no funcione correctamente debido al orden de las rutas.
    La ruta '/:id/configure' debería definirse ANTES de '/:pacienteId'.
    """
    url = f"{API_BASE_URL}/{specialist_id}/configure"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()
