
import sys
import os
sys.path.append(os.getcwd())

from langchain_core.messages import HumanMessage
from agents.lead_qualifier import lead_qualifier_node

def reproduce_issue():
    print("Testing Lead Qualifier with 'info del consultorio'...")
    
    state = {
        "messages": [HumanMessage(content="info del consultorio")],
        "intention": "",
        "next_node": ""
    }
    
    try:
        result = lead_qualifier_node(state)
        print("\n--- Result ---")
        print(result)
        
        # Check if the text matches the user's report
        if "messages" in result:
            msg_content = result["messages"][0].content
            print(f"\nResponse Content: {msg_content}")
            if "I'm sorry, but without any information provided" in msg_content:
                print("\n[MATCH] Reproduced the user's reported response.")
            else:
                print("\n[NO MATCH] Response is different.")
        else:
            print(f"\nIntention detected: {result.get('intention')}")
            print(f"Next Node: {result.get('next_node')}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    reproduce_issue()
