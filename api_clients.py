# api_clients.py

import requests
from typing import Dict, Any, Optional, List

# --- Configuración de la API ---
# Modifica esta URL para que apunte a la dirección base de tu API de citas.
# La he deducido del código del router de Express que proporcionaste.
API_BASE_URL = "http://localhost:3000/api/appointments"

def get_availability(
    empresa_id: str,
    especialista_id: str,
    fecha: str,
    paciente_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Llama a la API para obtener la disponibilidad de un especialista.
    Corresponde a: GET /
    """
    params = {
        "empresaId": empresa_id,
        "especialistaId": especialista_id,
        "fecha": fecha,
    }
    if paciente_id:
        params["pacienteId"] = paciente_id

    response = requests.get(API_BASE_URL, params=params)
    response.raise_for_status()  # Lanza una excepción para errores HTTP (4xx o 5xx)
    return response.json()

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
