from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import Tool
from datetime import datetime, timedelta

# Herramienta para guardar datos en un archivo de texto
# Usando librerias clasicas de langchain
def save_to_file(content: str, filename: str = "output.txt") -> str:
    # Guarda el contenido en un archivo de texto y devuelve el nombre del archivo."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    formatted_text = f"---\n# Guardado el {timestamp}\n---\n\n{content}"
    with open(filename, "a", encoding="utf-8") as file:
        file.write(formatted_text + "\n")
    return f"Contenido guardado en {filename}"

save_tool = Tool(
    name="save_data_to_file",
    func=save_to_file,
    description="Guarda el contenido proporcionado en un archivo de texto."
)

# Herramienta para obtener la hora actual
# usando el decorador @tool para definir una herramienta simple
from langchain.tools import tool
@tool
def gets_time() -> str:
    """Devuelve la hora y fecha actual en formato legible.""" 
    # implementar este docstring es sumante clave para entender la funcion de la herramienta
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"---\n# Hoy es {timestamp}\n"

horarios = { "lunes" : [{"inicio": "09:00", "fin": "13:00"}, {"inicio": "14:00", "fin": "17:00"}],
             "martes" : [{"inicio": "09:00", "fin": "13:00"}, {"inicio": "14:00", "fin": "17:00"}],
             "miercoles" : [{"inicio": "09:00", "fin": "13:00"}, {"inicio": "14:00", "fin": "17:00"}],
             "jueves" : [{"inicio": "09:00", "fin": "13:00"}, {"inicio": "14:00", "fin": "17:00"}],
             "viernes" : [{"inicio": "09:00", "fin": "13:00"}, {"inicio": "14:00", "fin": "17:00"}],
             "sabado" : [{"inicio": "09:00", "fin": "13:00"}, {"inicio": "14:00", "fin": "17:00"}],
             "domingo" : [] }

configura_citas = {
    "espacio_entre_citas": 5,  # minutos
    "duracion_cita": 30      # minutos
}

# funcion para generar horas disponibles
def generar_horas_disponibles(inicio: datetime, fin: datetime, duracion_cita: int, espacio_entre_citas: int):
    horas_disponibles = []
    hora_actual = inicio
    hora_fin = fin
    delta_cita = timedelta(minutes=duracion_cita + espacio_entre_citas)

    while hora_actual + timedelta(minutes=duracion_cita) <= hora_fin:
        horas_disponibles.append(hora_actual.strftime("%H:%M"))
        hora_actual += delta_cita

    return horas_disponibles

# Herramientas para devolver la disponibilidad de citas
@tool
def get_disponibilidad_citas(fecha: str) -> str:
    """Devuelve la disponibilidad de citas para la fecha proporcionada."""
    return {"lunes": [{"inicio": "09:00", "fin": "09:35"}, {"inicio": "10:10", "fin": "10:45"}, {"inicio": "11:20", "fin": "11:55"}, {"inicio": "14:00", "fin": "14:35"}, {"inicio": "15:10", "fin": "15:45"}, {"inicio": "16:20", "fin": "16:55"}],
            "martes": [{"inicio": "09:00", "fin": "09:35"}, {"inicio": "10:10", "fin": "10:45"}, {"inicio": "11:20", "fin": "11:55"}, {"inicio": "14:00", "fin": "14:35"}, {"inicio": "15:10", "fin": "15:45"}, {"inicio": "16:20", "fin": "16:55"}],
            "miercoles": [{"inicio": "09:00", "fin": "09:35"}, {"inicio": "10:10", "fin": "10:45"}, {"inicio": "11:20", "fin": "11:55"}, {"inicio": "14:00", "fin": "14:35"}, {"inicio": "15:10", "fin": "15:45"}, {"inicio": "16:20", "fin": "16:55"}],
            "jueves": [{"inicio": "09:00", "fin": "09:35"}, {"inicio": "10:10", "fin": "10:45"}, {"inicio": "11:20", "fin": "11:55"}, {"inicio": "14:00", "fin": "14:35"}, {"inicio": "15:10", "fin": "15:45"}, {"inicio": "16:20", "fin": "16:55"}],
            "viernes": [{"inicio": "09:00", "fin": "09:35"}, {"inicio": "10:10", "fin": "10:45"}, {"inicio": "11:20", "fin": "11:55"}, {"inicio": "14:00", "fin": "14:35"}, {"inicio": "15:10", "fin": "15:45"}, {"inicio": "16:20", "fin": "16:55"}],
            "sabado": [{"inicio": "09:00", "fin": "09:35"}, {"inicio": "10:10", "fin": "10:45"}, {"inicio": "11:20", "fin": "11:55"}],
            "domingo": [] }
