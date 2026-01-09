
import sys

print(f"Python executable: {sys.executable}")
print(f"Path: {sys.path}")

try:
    import langchain_chroma
    print(f"langchain_chroma module found: {langchain_chroma}")
    # print(f"in it: {dir(langchain_chroma)}")
    from langchain_chroma import Chroma
    print("Successfully imported Chroma from langchain_chroma")
except Exception as e:
    print(f"Error importing from langchain_chroma: {e}")

try:
    from langchain_community.vectorstores import Chroma
    print("Successfully imported Chroma from langchain_community.vectorstores (Fallback)")
except Exception as e:
    print(f"Error importing from langchain_community: {e}")
