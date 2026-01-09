
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from workflows.conversation_graph import create_app


# Load env vars (assuming .env exists or vars are set)
load_dotenv()

def run_interaction(app, config, message_content):
    print(f"\n--- User: {message_content} ---")
    inputs = {"messages": [HumanMessage(content=message_content)]}
    try:
        for event in app.stream(inputs, config, stream_mode="values"):
            messages = event.get("messages", [])
            state = event
            if messages:
                last_msg = messages[-1]
                print(f"[{last_msg.type.upper()}]: {last_msg.content[:100]}...")
    except Exception as e:
        print(f"!!! CRASH !!!: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    app = create_app()
    thread_id = "test-crash-repro-1"
    config = {"configurable": {"thread_id": thread_id}}

    # Turn 1
    run_interaction(app, config, "quiero agendar una cita")

    # Turn 2
    run_interaction(app, config, "muestrame la disponibilidad de hoy")
