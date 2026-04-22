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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

sessions = {}

def save_uploaded_file(uploaded_file: UploadFile) -> str:
    suffix = os.path.splitext(uploaded_file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.file.read())
        return tmp.name

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

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/summary")
async def generate_summary(document_id: str = Form(...)):
    rag = sessions[document_id]
    result = rag.summarize_contract_with_sources()
    return result

@app.post("/risk")
async def analyze_risk(document_id: str = Form(...)):
    rag = sessions[document_id]
    result = rag.analyze_risks()
    return result

@app.post("/ask")
async def ask_question(
    document_id: str = Form(...),
    question: str = Form(...)
):
    rag = sessions[document_id]
    result = rag.answer_question_with_sources(question)
    return result