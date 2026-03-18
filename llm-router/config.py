from dotenv import load_dotenv
import os

load_dotenv()

API_KEY        = os.getenv("API_KEY", "mypassword123")
GROQ_API_KEY   = os.getenv("GROQ_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
LOCAL_LLM_URL  = os.getenv("LOCAL_LLM_URL", "http://localhost:11434")
LOCAL_MODEL    = os.getenv("LOCAL_MODEL", "phi3:mini")

GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama3-8b-8192",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
]
GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-1.5-flash",
    "gemini-pro",
]
