from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import os
import uuid
import traceback

from contract_rag import ContractRAG

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://decoder-bot.lovable.app",
        "http://localhost:5173",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

sessions = {}


def save_uploaded_file(uploaded_file: UploadFile) -> str:
    if not uploaded_file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    suffix = os.path.splitext(uploaded_file.filename)[1]

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.file.read())
        return tmp.name


def get_rag_or_404(document_id: str) -> ContractRAG:
    rag = sessions.get(document_id)

    if rag is None:
        raise HTTPException(
            status_code=404,
            detail="Document not found. Please upload and process the PDF again."
        )

    return rag


@app.get("/")
def health_check():
    return {"status": "ok", "message": "AI NDA Analyzer API is running"}


@app.post("/process")
async def process_contract(file: UploadFile = File(...)):
    try:
        file_path = save_uploaded_file(file)

        rag = ContractRAG()
        num_chunks = rag.ingest_pdf(file_path)

        document_id = str(uuid.uuid4())
        sessions[document_id] = rag

        return {
            "document_id": document_id,
            "num_chunks": num_chunks,
            "filename": file.filename
        }

    except HTTPException:
        raise

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/summary")
async def generate_summary(document_id: str = Form(...)):
    try:
        rag = get_rag_or_404(document_id)
        return rag.summarize_contract_with_sources()

    except HTTPException:
        raise

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/risk")
async def analyze_risk(document_id: str = Form(...)):
    try:
        rag = get_rag_or_404(document_id)
        return rag.analyze_risks()

    except HTTPException:
        raise

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask")
async def ask_question(
    document_id: str = Form(...),
    question: str = Form(...)
):
    try:
        if not question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty.")

        rag = get_rag_or_404(document_id)
        return rag.answer_question_with_sources(question)

    except HTTPException:
        raise

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))