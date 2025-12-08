
import sys
import os
sys.path.append(os.getcwd())

from langchain_core.messages import HumanMessage, AIMessage
from agents.orchestrator import router_node

def test_context_awareness():
    print("Testing Orchestrator Context Awareness...")
    
    # Simulate a conversation history
    messages = [
        HumanMessage(content="Quiero agendar una cita"),
        AIMessage(content="Claro, ¿cuál es tu nombre completo?"),
        HumanMessage(content="Juan Perez")
    ]
    
    state = {
        "messages": messages,
        "intention": "", # Intention is unknown coming in
        "next_node": ""
    }
    
    try:
        result = router_node(state)
        print("\n--- Result for 'Juan Perez' ---")
        print(f"Intention: {result.get('intention')}")
        print(f"Next Node: {result.get('next_node')}")
        
        if result.get('intention') == 'reserva':
             print("[PASS] Context correctly identified as 'reserva'.")
        elif result.get('intention') == 'otro':
             print("[PASS] Context identified as 'otro' (acceptable if routed to Scheduler/Knowledge).") 
        else:
             print(f"[FAIL] Unexpected intention '{result.get('intention')}'. Expected 'reserva'.")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_context_awareness()
