from dotenv import load_dotenv
import os
load_dotenv()

API_KEY        = os.getenv("API_KEY", "mypassword123")
GROQ_API_KEY   = os.getenv("GROQ_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
LOCAL_LLM_URL  = os.getenv("LOCAL_LLM_URL", "http://localhost:11434")
LOCAL_MODEL    = os.getenv("LOCAL_MODEL", "phi3:mini")
MAX_HISTORY    = 12
