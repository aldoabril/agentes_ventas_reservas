
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from config import GEMINI_MODELS

load_dotenv()

def run_gemini_test(name, messages):
    print(f"\n--- TEST: {name} ---")
    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODELS.GEMINI_25_FLASH.value,
        temperature=0
    )
    try:
        response = llm.invoke(messages)
        print("SUCCESS")
    except Exception as e:
        print(f"FAILED: {e}")

if __name__ == "__main__":
    # Setup valid exchange
    # User -> AI(TC) -> Tool -> AI(Text) -> User
    
    msg_user_1 = HumanMessage(content="Info de Dr. Juan")
    msg_ai_tc = AIMessage(content="", tool_calls=[{"name": "get_availability", "args": {}, "id": "123"}])
    msg_tool = ToolMessage(content="Disponibilidad: 9am", tool_call_id="123", name="get_availability")
    msg_ai_text = AIMessage(content="Tiene a las 9am")
    msg_user_2 = HumanMessage(content="Ok agendar")

    full_history = [msg_user_1, msg_ai_tc, msg_tool, msg_ai_text, msg_user_2]

    # Case 1: Start with ToolMessage (Slice skips User and AI_TC)
    # History: [Tool, AI, User]
    slice_1 = full_history[2:] 
    print(f"Slice 1 types: {[type(m).__name__ for m in slice_1]}")
    run_gemini_test("Start with ToolMessage", slice_1)

    # Case 2: Start with AI ToolCall (Slice skips User)
    # History: [AI(TC), Tool, AI, User]
    slice_2 = full_history[1:]
    print(f"Slice 2 types: {[type(m).__name__ for m in slice_2]}")
    run_gemini_test("Start with AI ToolCall", slice_2)

    # Case 3: Proper start (User)
    slice_3 = full_history[0:]
    run_gemini_test("Start with User", slice_3)
