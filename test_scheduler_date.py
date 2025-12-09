
import sys
import os
sys.path.append(os.getcwd())

from langchain_core.messages import HumanMessage, AIMessage
from agents.scheduler import scheduler_node
from datetime import datetime, timedelta

def test_past_date_rejection():
    print("Testing Scheduler Past Date Rejection...")
    
    # Calculate a date in the past (yesterday)
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    
    # Simulate state: User wants to book for yesterday
    messages = [
        HumanMessage(content=f"Quiero agendar una cita para el {yesterday}")
    ]
    
    state = {
        "messages": messages,
        "intention": "reserva", 
        "next_node": "Scheduler"
    }
    
    print(f"User Input: Quiero agendar una cita para el {yesterday}")
    
    try:
        result = scheduler_node(state)
        ai_responses = [m for m in result['messages'] if isinstance(m, AIMessage) or hasattr(m, 'tool_calls')]
        
        last_response = ai_responses[-1]
        
        print("\n--- Scheduler Response ---")
        if last_response.tool_calls:
            print(f"Tool Calls: {last_response.tool_calls}")
            print("[WARNING] Scheduler attempted to call a tool. Check if it's checking availability or just failing.")
            # If it calls get_availability, it might be checking IS valid or not. 
            # But the prompt says "No se puede...". Ideally it should refuse.
        else:
            print(f"Content: {last_response.content}")
            if "anterior" in last_response.content.lower() or "pasado" in last_response.content.lower() or "no se puede" in last_response.content.lower():
                 print("[PASS] Scheduler refused the past date.")
            else:
                 print("[UNCERTAIN] Check content for refusal.")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_past_date_rejection()
