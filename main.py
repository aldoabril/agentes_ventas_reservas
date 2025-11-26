"""
Aplicación principal del sistema de agentes de ventas y reservas.
"""
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from workflows.conversation_graph import create_app

# Cargar variables de entorno
load_dotenv()

if __name__ == "__main__":
    # Crear la aplicación del grafo
    app = create_app()
    
    # Configuración de threading para memoria conversacional
    thread_id = "mi-conversacion-1"
    config = {"configurable": {"thread_id": thread_id}}
    
    print("🤖 Hola, soy tu asistente. Escribe 'salir' para terminar.")
    
    while True:
        user_input = input("🙂 Tú: ")
        if user_input.lower() in ["salir", "exit"]:
            break
        
        inputs = {"messages": [HumanMessage(content=user_input)]}
        
        for event in app.stream(inputs, config, stream_mode="values"):
            event["messages"][-1].pretty_print()
