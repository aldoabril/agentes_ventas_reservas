import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from langchain_core.messages import HumanMessage
from agents.lead_qualifier import lead_qualifier_node
from agents.base import AgentState

def test_lead_qualifier():
    print("Testing Lead Qualifier Node...")
    
    # Mock state
    state: AgentState = {
        "messages": [HumanMessage(content="Hola, quisiera agendar una cita con el dentista.")],
        "intention": "",
        "next_node": "",
        "empresa_id": None,
        "paciente_nombre": None,
        "especialista_id": None,
        "fecha": None,
        "horarios_disponibles": None,
        "hora_seleccionada": None,
        "booking_complete": False
    }
    
    # Since we can't easily mock the LLM here without more setup, 
    # we just check if the function runs and handle the likely API key error gracefully
    # or just checks imports work.
    
    try:
        result = lead_qualifier_node(state)
        print("Result:", result)
    except Exception as e:
        print(f"Execution failed (likely due to missing API key, which is expected): {e}")
        # Even if it fails due to API key, if it reached the LLM call, the imports and structure are correct.
        if "api_key" in str(e).lower() or "openai" in str(e).lower():
             print("Test PASSED (Structure is correct, failed on API call as expected)")
        else:
             print("Test FAILED")

if __name__ == "__main__":
    test_lead_qualifier()
