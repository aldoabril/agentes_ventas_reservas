import os
from dotenv import load_dotenv
load_dotenv()

print(f"Tracing: {os.getenv('LANGCHAIN_TRACING_V2')}")
print(f"Project: {os.getenv('LANGCHAIN_PROJECT')}")
print(f"API Key: {os.getenv('LANGCHAIN_API_KEY', '')[:15]}...")

from langsmith import traceable

@traceable(name="test-conexion")
def mi_test():
    return "¡Hola LangSmith!"

resultado = mi_test()
print(f"Resultado: {resultado}")
print("✅ Ahora revisa LangSmith en 10-15 segundos")