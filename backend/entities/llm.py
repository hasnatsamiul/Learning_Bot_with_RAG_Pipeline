
import os
from langchain_google_genai import ChatGoogleGenerativeAI

GOOGLE_MODEL = os.getenv("GOOGLE_MODEL", "gemini-2.5-flash")

llm = ChatGoogleGenerativeAI(
    model=GOOGLE_MODEL,
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)