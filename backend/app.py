from dotenv import load_dotenv
load_dotenv()

import os
import json
import base64
import traceback
from typing import TypedDict, Optional, List, Annotated

from fastapi import FastAPI, HTTPException, Response, status, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import MessagesState, StateGraph
from langgraph.prebuilt import create_react_agent
from langchain_community.vectorstores import FAISS

from entities.variables import AGENT_NAME, TOP_K, ALLOWED_ORIGINS, ORG_NAME, FAISS_INDEX_DIR
from entities.embedder import embedder
from entities.llm import llm


# FastAPI
app = FastAPI()
security = HTTPBearer(auto_error=False)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DISABLE_AUTH = os.getenv("DISABLE_AUTH", "false").strip().lower() == "true"
CLERK_PEM_PUBLIC_KEY = os.getenv("CLERK_PEM_PUBLIC_KEY")  # optional, only needed if auth enabled


# Load FAISS index

try:
    vectorstore = FAISS.load_local(
        FAISS_INDEX_DIR,
        embeddings=embedder,
        allow_dangerous_deserialization=True,
    )
    print("FAISS index loaded.")
except Exception as e:
    print("Failed to load FAISS index:", e)
    raise

# Schemas

class Payload(BaseModel):
    chat_history: List[dict]

class ResponseSchema(TypedDict):
    response: str
    source_pdfs: Annotated[Optional[List[str]], ..., "List of source PDF names"]
    source_images: Annotated[Optional[List[str]], ..., "List of source image paths"]


# Tool: Vector search

@tool
def query_vectorstore(query: str):
    """Similarity search on the FAISS vectorstore."""
    return vectorstore.similarity_search(query=query, k=TOP_K)


# RAG Agent Node

def rag_agent_node():
    system_prompt = f"""
You are a retrieval-augmented generation (RAG) agent for {ORG_NAME}.

Your job is to ONLY answer questions based on context retrieved from the 'query_vectorstore' tool.

Rules:
- ALWAYS call the 'query_vectorstore' tool for any user question.
- If needed, rephrase the query and call the tool multiple times to get better results.
- If no answer is found in the retrieved context, respond with: "I don't know."
- Output MUST ALWAYS be JSON with keys: `response`, `source_pdfs`, and `source_images`.
- Put all relevant source PDF names in `source_pdfs`.
- Put all relevant image file paths in `source_images`.
- Do NOT fabricate or guess.
- Do NOT use outside knowledge.
- For basic greetings, reply briefly WITHOUT calling the tool and keep sources empty.
""".strip()

    return create_react_agent(
        model=llm,
        tools=[query_vectorstore],
        name=AGENT_NAME,
        prompt=system_prompt,
    )


# Structured Response Node

def structured_response_agent(state: MessagesState):
    response_obj = llm.with_structured_output(ResponseSchema).invoke([
        SystemMessage(content="""
You are a structured response agent.
Convert the model output into a structured JSON response:
Keys: response, source_pdfs, source_images.
If it's just a greeting, keep source_pdfs and source_images empty.
""".strip()),
        HumanMessage(content=state["messages"][-1].content),
    ])

    res_json = {
        "response": (response_obj.get("response") or ""),
        "source_pdfs": (response_obj.get("source_pdfs") or []) or [],
        "source_images": (response_obj.get("source_images") or []) or [],
    }

    return {"messages": [AIMessage(content=json.dumps(res_json), name="structured_response_agent")]}


# Optional auth

def verify_token(credentials: HTTPAuthorizationCredentials | None, response: Response):
    if DISABLE_AUTH:
        return  # skip

    if credentials is None:
        response.status_code = status.HTTP_401_UNAUTHORIZED
        raise HTTPException(status_code=401, detail="Missing token")

    if not CLERK_PEM_PUBLIC_KEY:
        raise HTTPException(status_code=500, detail="CLERK_PEM_PUBLIC_KEY is not set")

    try:
        import jwt
        jwt.decode(credentials.credentials, key=CLERK_PEM_PUBLIC_KEY, algorithms=["RS256"])
    except Exception:
        response.status_code = status.HTTP_401_UNAUTHORIZED
        raise HTTPException(status_code=401, detail="Invalid token")


# Endpoint

@app.post("/query")
async def call_agent(
    req: Payload,
    response: Response,
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
):
    try:
        verify_token(credentials, response)

        agent = rag_agent_node()

        workflow = StateGraph(MessagesState)
        workflow.add_node("rag_agent", agent)
        workflow.add_node("structured_response_agent", structured_response_agent)
        workflow.add_edge("__start__", "rag_agent")
        workflow.add_edge("rag_agent", "structured_response_agent")
        graph = workflow.compile()

        out = graph.invoke({"messages": req.chat_history})
        res = json.loads(out["messages"][-1].content)

        # Convert image paths to base64 strings (if any)
        if res.get("source_images"):
            res["source_images"] = [
                base64.b64encode(open(path, "rb").read()).decode("utf-8")
                for path in res["source_images"]
                if isinstance(path, str) and os.path.exists(path)
            ]

        return {"result": res}

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))