# AI NDA Analyzer

An AI-powered contract review assistant for Non-Disclosure Agreements (NDAs).

This project uses **OpenAI GPT**, **RAG (Retrieval-Augmented Generation)**, **ChromaDB**, and **Streamlit** to help users understand contracts faster through summarization, risk detection, and clause-based Q&A.

---

## Features

### Smart Contract Summary
Upload an NDA PDF and receive a plain-English summary including:

- Parties involved
- Effective date
- Confidentiality obligations
- Term / duration
- Governing law
- Return / destruction obligations
- Important unusual clauses

---

### Risk Analysis

Hybrid risk engine using:

- Rule-based logic
- GPT reasoning

Examples of risks detected:

- Non-compete clauses
- Long confidentiality terms
- Broad confidential information definitions
- Missing exclusions
- One-sided obligations

---

### Contract Q&A

Ask natural language questions such as:

- How long does confidentiality last?
- Is there a non-compete clause?
- What law governs this agreement?
- Can information be shared with affiliates?

---

### Explainable AI Outputs

Each answer includes supporting evidence:

- Page number
- Clause title
- Source text snippet

This helps reduce hallucination and improve trust.

---

## Tech Stack

- Python
- Streamlit
- OpenAI API
- LangChain
- ChromaDB
- OCR (RapidOCR / PyMuPDF)
- Vector Embeddings

---

## System Architecture

```text
User uploads PDF
      ↓
PDF Parsing + OCR
      ↓
Clause-aware Chunking
      ↓
Embeddings
      ↓
Vector Database
      ↓
OpenAI GPT Analysis
      ↓
Dashboard Output
