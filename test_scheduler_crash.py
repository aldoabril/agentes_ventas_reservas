
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from agents.scheduler import scheduler_node


load_dotenv()

def run_test(name, messages):
    print(f"\n--- TEST: {name} ---")
    state = {"messages": messages}
    try:
        scheduler_node(state)
        print("SUCCESS")
    except Exception as e:
        print(f"FAILED: {e}")

if __name__ == "__main__":
    # Case 1: Simple User Input
    run_test("Simple User", [HumanMessage(content="muestrame la disponibilidad de hoy")])

    # Case 2: User -> AI -> User
    run_test("Conversation", [
        HumanMessage(content="quiero agendar"),
        AIMessage(content="Hola, nombre?"),
        HumanMessage(content="muestrame la disponibilidad de hoy")
    ])

    # Case 3: User -> AI (Text) -> AI (Text) [Simulating missing User input]
    run_test("Missing User Input", [
        HumanMessage(content="quiero agendar"),
        AIMessage(content="Hola, nombre?")
    ])
    
    # Case 4: User -> AI (ToolCall) [Simulating interrupted tool loop]
    # Note: AI with tool_calls must be followed by ToolMessage, but here we simulate 'missing' ToolMessage
    msg_with_tool = AIMessage(content="", tool_calls=[{"name": "foo", "args": {}, "id": "123"}])
    run_test("Interrupted Tool Call", [
        HumanMessage(content="quiero agendar"),
        msg_with_tool
    ])
    
    # Case 5: User -> AI (Text) -> AI (ToolCall)
    # We provide all info so model attempts to call tool, but we insert an AI message before it.
    run_test("AI Text then ToolCall", [
        HumanMessage(content="Soy Aldo, quiero una cita para hoy. USA LA HERRAMIENTA get_availability YA."),
        AIMessage(content="Entendido, voy a verificar la disponibilidad usando la herramienta.") 
    ])
