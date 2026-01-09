
import sys
import os
sys.path.append(os.getcwd())

from langchain_core.messages import HumanMessage, AIMessage
from agents.lead_qualifier import lead_qualifier_node

def test_context_awareness():
    print("Testing Lead Qualifier Context Awareness...")
    
    # Simulate a conversation history
    # 1. User asks for appointment
    # 2. Bot asks for name
    # 3. User responds with name "Juan" -> This should be classified as 'reserva' or 'otro', NOT 'invalido'
    
    messages = [
        HumanMessage(content="Quiero agendar una cita"),
        AIMessage(content="Claro, ¿cuál es tu nombre completo?"),
        HumanMessage(content="Juan Perez")
    ]
    
    state = {
        "messages": messages,
        "intention": "",
        "next_node": ""
    }
    
    try:
        result = lead_qualifier_node(state)
        print("\n--- Result for 'Juan Perez' ---")
        print(f"Intention: {result.get('intention')}")
        print(f"Next Node: {result.get('next_node')}")
        
        if result.get('intention') == 'reserva':
             print("[PASS] Context correctly identified as 'reserva'.")
        elif result.get('intention') == 'otro':
             print("[PASS] Context identified as 'otro' (acceptable).") # 'otro' is often safe fallback
        else:
             print(f"[FAIL] Unexpected intention '{result.get('intention')}'. Expected 'reserva'.")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_context_awareness()
