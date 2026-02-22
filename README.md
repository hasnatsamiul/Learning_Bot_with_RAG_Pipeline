# Learning_Bot_RAG_Pipeline

A Retrieval-Augmented Generation (RAG) pipeline built using **FastAPI, LangGraph, FAISS, HuggingFace Embeddings, and Google Gemini (GenAI)**.

This project enables users to query PDF documents and receive grounded, context-aware answers powered by semantic search and LLM reasoning.

---

##  What This Project Does

- Loads and processes PDF documents
- Converts text into vector embeddings
- Stores embeddings in FAISS
- Retrieves relevant context using similarity search
- Uses Gemini LLM to generate structured responses
- Returns JSON output with answer and source references

The system ensures answers are based only on indexed documents.


<img width="1507" height="829" alt="Screenshot 2026-02-21 at 23 03 13" src="https://github.com/user-attachments/assets/fcb852d4-90b9-41bf-80bb-bec54419dcd8" />


---

##  Features

- FAISS vector similarity search
- LangGraph-based RAG agent
- FastAPI backend (`POST /query`)
- PDF indexing workflow
- Structured JSON output (`response`, `source_pdfs`, `source_images`)

---


---

##  Setup (Backend)

```bash
cd backend
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt

# Create environment file
cp .env.example .env

#Open .env and add your Gemini API key:
GOOGLE_API_KEY=YOUR_KEY_HERE
GOOGLE_MODEL=gemini-2.5-flash
#Place your PDF files inside:
backend/pdfs/
#Then Run
python source_indexing.py
#Run the server
uvicorn app:app --reload --port 8000
```

##  Setup (Frontend)
```
cd frontend
npm install
npm run dev
```

### This will:

- Load PDFs

- Generate embeddings

- Create the FAISS index

- Store vectors in faiss_index/
- RAG ensures answers are based only on indexed documents

### If you find any problem contact 
smhasnats@gmail.com
