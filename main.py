"""
Aplicación principal del sistema de agentes de ventas y reservas.
"""

import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from workflows.conversation_graph import create_app
from agents.memory import truncate_messages

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
            # Mantener una ventana corta de mensajes en memoria para minimizar contexto
            # (últimas 5 por defecto). Esto evita que el estado crezca indefinidamente.
            try:
                event_messages = event.get("messages", None)
                truncated = truncate_messages(event_messages)
                if truncated is not None:
                    event["messages"] = truncated
            except Exception:
                # No interrumpir el flujo por errores de truncado
                pass

            event["messages"][-1].pretty_print()
