from __future__ import annotations

import logging
import os
import re
import tempfile
import uuid
from typing import Dict, List, Tuple

import pymupdf
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rapidocr_onnxruntime import RapidOCR

from prompts import (
    CONFIDENTIALITY_TERM_QA_PROMPT,
    FIELD_EXTRACTION_PROMPT,
    QA_PROMPT,
    RISK_ANALYSIS_PROMPT,
    RISK_QA_PROMPT,
    SUMMARY_PROMPT,
)
from retrieval_queries import (
    CONFIDENTIALITY_TERM_RETRIEVAL_QUERY,
    GENERAL_SUMMARY_RETRIEVAL_QUERY,
    KEY_FIELDS_FALLBACK_RETRIEVAL_QUERY,
    RISK_ANALYSIS_RETRIEVAL_QUERY,
    RISK_QA_RETRIEVAL_QUERY,
)

logger = logging.getLogger(__name__)
NOT_FOUND = "Not found"


class ContractRAG:
    def __init__(
        self,
        llm_model: str = "gpt-4o-mini",
        embedding_model: str = "text-embedding-3-small",
        persist_directory: str | None = None,
    ) -> None:
        self.llm_model = llm_model
        self.embedding_model = embedding_model

        if persist_directory is None:
            persist_directory = os.path.join("data", f"chroma_db_{uuid.uuid4().hex}")

        self.persist_directory = persist_directory
        os.makedirs(self.persist_directory, exist_ok=True)

        self.llm = ChatOpenAI(model=self.llm_model, temperature=0)
        self.embeddings = OpenAIEmbeddings(model=self.embedding_model)

        self.vectorstore = None
        self.raw_docs: List[Document] = []
        self.ocr_engine = RapidOCR()

    # ----------------------------
    # PDF ingestion with OCR fallback
    # ----------------------------
    def load_pdf(self, pdf_path: str) -> List[Document]:
        loader = PyPDFLoader(pdf_path)
        base_docs = loader.load()

        pdf_doc = pymupdf.open(pdf_path)
        final_docs: List[Document] = []

        for page_number, doc in enumerate(base_docs):
            page_text = doc.page_content or ""

            if self._needs_ocr_fallback(page_text):
                ocr_text = self._ocr_page(pdf_doc, page_number)
                chosen_text = self._merge_text(page_text, ocr_text)
            else:
                chosen_text = page_text

            cleaned_text = self._clean_text(chosen_text)

            final_docs.append(
                Document(
                    page_content=cleaned_text,
                    metadata=doc.metadata,
                )
            )

        pdf_doc.close()
        return final_docs

    def _needs_ocr_fallback(self, text: str) -> bool:
        if not text.strip():
            return True

        if len(text.strip()) < 30:
            return True

        return False

    def _ocr_page(self, pdf_doc: pymupdf.Document, page_number: int) -> str:
        page = pdf_doc.load_page(page_number)
        matrix = pymupdf.Matrix(2.0, 2.0)
        pix = page.get_pixmap(matrix=matrix, alpha=False)

        fd, image_path = tempfile.mkstemp(suffix=".png")
        os.close(fd)

        try:
            pix.save(image_path)

            ocr_result, _ = self.ocr_engine(image_path)

            if not ocr_result:
                return ""

            lines: List[str] = []
            for item in ocr_result:
                if len(item) >= 2 and isinstance(item[1], str):
                    lines.append(item[1])

            return "\n".join(lines).strip()

        finally:
            try:
                if os.path.exists(image_path):
                    os.remove(image_path)
            except PermissionError:
                logger.warning("Could not remove temp OCR file: %s", image_path)

    def _merge_text(self, base_text: str, ocr_text: str) -> str:
        base_text = (base_text or "").strip()
        ocr_text = (ocr_text or "").strip()

        if not ocr_text:
            return base_text
        if not base_text:
            return ocr_text

        return f"{base_text}\n\n[OCR]\n{ocr_text}"

    def _clean_text(self, text: str) -> str:
        text = text.replace("\x00", " ")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"([a-z])-\n([a-z])", r"\1\2", text)
        text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
        text = re.sub(r"\s{2,}", " ", text)
        return text.strip()

    # ----------------------------
    # Clause-aware-ish splitting
    # ----------------------------
    def split_documents(self, docs: List[Document]) -> List[Document]:
        clause_docs: List[Document] = []

        for doc in docs:
            page = doc.metadata.get("page", "unknown")
            page_text = doc.page_content

            sections = self._split_by_section_headers(page_text)

            if not sections:
                clause_docs.append(
                    Document(
                        page_content=page_text,
                        metadata={**doc.metadata, "clause_title": "Page Text"},
                    )
                )
                continue

            for title, content in sections:
                clause_docs.append(
                    Document(
                        page_content=content,
                        metadata={
                            **doc.metadata,
                            "clause_title": title,
                            "source_page": page,
                        },
                    )
                )

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=150,
        )

        split_docs: List[Document] = []
        for doc in clause_docs:
            if len(doc.page_content) <= 1400:
                split_docs.append(doc)
            else:
                split_docs.extend(splitter.split_documents([doc]))

        return split_docs

    def _split_by_section_headers(self, text: str) -> List[Tuple[str, str]]:
        patterns = [
            r"(?=(?:^|\n)\s*(?:Section|SECTION)\s+\d+[A-Za-z0-9.\-]*\s*[:.\-]?\s*[A-Z][^\n]{0,120})",
            r"(?=(?:^|\n)\s*\d+(?:\.\d+)*\s+[A-Z][^\n]{0,120})",
            r"(?=(?:^|\n)\s*(?:Confidential Information|Exclusions|Term|Termination|Return of Materials|Remedies|Governing Law|Purpose|Non[- ]Solicitation|Non[- ]Compete|Injunctive Relief|Assignment)[^\n]{0,120})",
        ]

        split_positions = set()
        for pattern in patterns:
            for match in re.finditer(pattern, text, flags=re.IGNORECASE | re.MULTILINE):
                split_positions.add(match.start())

        if not split_positions:
            return []

        positions = sorted(split_positions)
        sections: List[Tuple[str, str]] = []

        for i, start in enumerate(positions):
            end = positions[i + 1] if i + 1 < len(positions) else len(text)
            chunk = text[start:end].strip()

            if not chunk:
                continue

            first_line = chunk.splitlines()[0].strip()
            title = first_line[:120] if first_line else "Untitled Clause"
            sections.append((title, chunk))

        return sections

    # ----------------------------
    # Vector store
    # ----------------------------
    def build_vectorstore(self, docs: List[Document]) -> None:
        self.vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
            collection_name="nda_contracts",
        )

    def ingest_pdf(self, pdf_path: str) -> int:
        self.raw_docs = self.load_pdf(pdf_path)
        split_docs = self.split_documents(self.raw_docs)
        self.build_vectorstore(split_docs)
        return len(split_docs)

    # ----------------------------
    # Retrieval helpers
    # ----------------------------
    def _retrieve_docs(self, query: str, k: int = 6) -> List[Document]:
        if self.vectorstore is None:
            raise ValueError("Vector store is not initialized. Upload and process a PDF first.")

        retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})
        return retriever.invoke(query)

    def _format_docs(self, docs: List[Document]) -> str:
        parts: List[str] = []
        for i, doc in enumerate(docs, start=1):
            page = doc.metadata.get("page", "unknown")
            clause_title = doc.metadata.get("clause_title", "Unknown Clause")
            parts.append(
                f"[Chunk {i} | Page {page} | Clause {clause_title}]\n{doc.page_content}"
            )
        return "\n\n".join(parts)

    def _docs_to_sources(self, docs: List[Document], max_chars: int = 400) -> List[Dict]:
        return [
            {
                "page": doc.metadata.get("page", "unknown"),
                "clause_title": doc.metadata.get("clause_title", "Unknown Clause"),
                "supporting_text": doc.page_content[:max_chars],
            }
            for doc in docs
        ]

    def _get_first_pages_text(self, max_pages: int = 2) -> str:
        if not self.raw_docs:
            raise ValueError("Raw documents are not loaded. Upload and process a PDF first.")

        selected: List[Document] = []
        for doc in self.raw_docs:
            page = doc.metadata.get("page", 0)
            if isinstance(page, int) and page < max_pages:
                selected.append(doc)

        return self._format_docs(selected)

    def _call_llm(self, prompt: str) -> str:
        response = self.llm.invoke(prompt)
        return response.content if hasattr(response, "content") else str(response)

    # ----------------------------
    # NDA key field extraction
    # ----------------------------
    def _empty_key_fields(self) -> Dict[str, str]:
        return {
            "contract_type": NOT_FOUND,
            "parties": NOT_FOUND,
            "effective_date": NOT_FOUND,
            "purpose_of_disclosure": NOT_FOUND,
            "confidential_information_definition": NOT_FOUND,
            "exclusions": NOT_FOUND,
            "confidentiality_term": NOT_FOUND,
            "return_or_destruction": NOT_FOUND,
            "governing_law": NOT_FOUND,
            "non_solicitation_or_non_compete": NOT_FOUND,
        }

    def _normalize_field_value(self, value: str) -> str:
        v = (value or "").strip()
        if not v:
            return NOT_FOUND
        if v.lower() in {"not found", "n/a", "none", "null", "unknown"}:
            return NOT_FOUND
        return v

    def _extract_key_fields_regex(self, text: str) -> Dict[str, str]:
        result = self._empty_key_fields()
        text_one_line = " ".join(text.split())

        if re.search(
            r"non[- ]disclosure agreement|nda|confidentiality agreement",
            text_one_line,
            flags=re.IGNORECASE,
        ):
            result["contract_type"] = "NDA / Confidentiality Agreement"

        effective_patterns = [
            r"(?:effective\s+date|dated\s+as\s+of|made\s+and\s+entered\s+into\s+as\s+of)\s*[:\-]?\s*([A-Za-z]+\s+\d{1,2},\s+\d{4})",
            r"this\s+(?:agreement|nda|confidentiality agreement)\s+is\s+made\s+as\s+of\s+([A-Za-z]+\s+\d{1,2},\s+\d{4})",
        ]
        for pattern in effective_patterns:
            match = re.search(pattern, text_one_line, flags=re.IGNORECASE)
            if match:
                result["effective_date"] = match.group(1).strip()
                break

        parties_patterns = [
            r"between\s+(.+?)\s+and\s+(.+?)(?:\.|,|\s+for\s+the\s+purpose)",
            r"by\s+and\s+between\s+(.+?)\s+and\s+(.+?)(?:\.|,)",
        ]
        for pattern in parties_patterns:
            match = re.search(pattern, text_one_line, flags=re.IGNORECASE)
            if match:
                p1 = match.group(1).strip()
                p2 = match.group(2).strip()
                result["parties"] = f"{p1} ; {p2}"
                break

        governing_law_match = re.search(
            r"governed\s+by\s+the\s+laws\s+of\s+([A-Za-z\s]+?)(?:\.|,|;)",
            text_one_line,
            flags=re.IGNORECASE,
        )
        if governing_law_match:
            result["governing_law"] = governing_law_match.group(1).strip()

        markers = []
        if re.search(r"non[- ]solicit|non[- ]solicitation", text_one_line, flags=re.IGNORECASE):
            markers.append("Non-solicitation language detected")
        if re.search(r"non[- ]compete", text_one_line, flags=re.IGNORECASE):
            markers.append("Non-compete language detected")
        if markers:
            result["non_solicitation_or_non_compete"] = "; ".join(markers)

        if re.search(
            r"return\s+or\s+destroy|destroy\s+all\s+copies|return\s+all\s+materials",
            text_one_line,
            flags=re.IGNORECASE,
        ):
            result["return_or_destruction"] = "Return / destruction obligation detected."

        term_patterns = [
            r"(?:for\s+a\s+period\s+of|for)\s+(\d+\s+(?:year|years|month|months))",
            r"survive\s+for\s+(\d+\s+(?:year|years|month|months))",
            r"confidentiality\s+obligations\s+.*?\s+for\s+(\d+\s+(?:year|years|month|months))",
        ]
        for pattern in term_patterns:
            match = re.search(pattern, text_one_line, flags=re.IGNORECASE)
            if match:
                result["confidentiality_term"] = match.group(1).strip()
                break

        return result

    def _extract_key_fields_llm(self, context: str) -> Dict[str, str]:
        prompt = FIELD_EXTRACTION_PROMPT.format(context=context)
        raw = self._call_llm(prompt)

        result = self._empty_key_fields()

        for line in raw.splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip()
            value = self._normalize_field_value(value)
            if key in result:
                result[key] = value

        return result

    def _merge_missing_fields(
        self,
        primary: Dict[str, str],
        fallback: Dict[str, str],
        keys: List[str],
    ) -> Dict[str, str]:
        merged = dict(primary)
        for key in keys:
            if merged.get(key, NOT_FOUND) == NOT_FOUND:
                merged[key] = fallback.get(key, NOT_FOUND)
        return merged

    def extract_key_fields_simple(self) -> Dict[str, str]:
        if not self.raw_docs:
            raise ValueError("Raw documents are not loaded. Upload and process a PDF first.")

        raw_context = self._get_first_pages_text(max_pages=2)

        raw_regex_result = self._extract_key_fields_regex(raw_context)
        raw_llm_result = self._extract_key_fields_llm(raw_context)

        merged_fields = dict(raw_regex_result)
        merged_fields = self._merge_missing_fields(
            merged_fields,
            raw_llm_result,
            keys=list(self._empty_key_fields().keys()),
        )

        missing_core = any(
            merged_fields[field] == NOT_FOUND
            for field in [
                "effective_date",
                "parties",
                "confidentiality_term",
                "governing_law",
            ]
        )

        if missing_core:
            retrieved_docs = self._retrieve_docs(
                query=KEY_FIELDS_FALLBACK_RETRIEVAL_QUERY,
                k=8,
            )
            retrieved_context = self._format_docs(retrieved_docs)

            retrieved_regex_result = self._extract_key_fields_regex(retrieved_context)
            retrieved_llm_result = self._extract_key_fields_llm(retrieved_context)

            merged_fields = self._merge_missing_fields(
                merged_fields,
                retrieved_regex_result,
                keys=list(self._empty_key_fields().keys()),
            )
            merged_fields = self._merge_missing_fields(
                merged_fields,
                retrieved_llm_result,
                keys=list(self._empty_key_fields().keys()),
            )

        return merged_fields

    def _format_key_fields_for_summary(self, fields: Dict[str, str]) -> str:
        return (
            "Extracted Key Fields:\n"
            f"- contract_type: {fields.get('contract_type', NOT_FOUND)}\n"
            f"- parties: {fields.get('parties', NOT_FOUND)}\n"
            f"- effective_date: {fields.get('effective_date', NOT_FOUND)}\n"
            f"- purpose_of_disclosure: {fields.get('purpose_of_disclosure', NOT_FOUND)}\n"
            f"- confidential_information_definition: {fields.get('confidential_information_definition', NOT_FOUND)}\n"
            f"- exclusions: {fields.get('exclusions', NOT_FOUND)}\n"
            f"- confidentiality_term: {fields.get('confidentiality_term', NOT_FOUND)}\n"
            f"- return_or_destruction: {fields.get('return_or_destruction', NOT_FOUND)}\n"
            f"- governing_law: {fields.get('governing_law', NOT_FOUND)}\n"
            f"- non_solicitation_or_non_compete: {fields.get('non_solicitation_or_non_compete', NOT_FOUND)}\n"
        )

    def _get_confidentiality_term_for_summary(self, fields: Dict[str, str]) -> str:
        term = fields["confidentiality_term"]

        if term != NOT_FOUND:
            return f"The confidentiality term is: {term}"

        docs = self._retrieve_docs(
            query=CONFIDENTIALITY_TERM_RETRIEVAL_QUERY,
            k=10,
        )
        context = self._format_docs(docs)
        prompt = CONFIDENTIALITY_TERM_QA_PROMPT.format(
            question="What is the confidentiality term?",
            context=context,
        )
        return self._call_llm(prompt)

    # ----------------------------
    # NDA risk scoring
    # ----------------------------
    def _score_nda_risks(self, docs: List[Document]) -> Dict:
        findings = []
        total_score = 0

        for doc in docs:
            text = doc.page_content
            page = doc.metadata.get("page", "unknown")
            clause_title = doc.metadata.get("clause_title", "Unknown Clause")

            clause_score = 0
            reasons = []

            long_term_match = re.search(r"(\d+)\s+(year|years)", text, flags=re.IGNORECASE)
            if long_term_match:
                years = int(long_term_match.group(1))
                if years > 5:
                    clause_score += 25
                    reasons.append(
                        f"Confidentiality obligation may last more than 5 years ({years} years)."
                    )

            if "exclusions" not in text.lower() and "shall not include" not in text.lower():
                clause_score += 20
                reasons.append("No clear exclusions from confidential information were detected.")

            if re.search(r"non[- ]compete", text, flags=re.IGNORECASE):
                clause_score += 35
                reasons.append("Non-compete language detected.")

            if re.search(r"non[- ]solicit|non[- ]solicitation", text, flags=re.IGNORECASE):
                clause_score += 20
                reasons.append("Non-solicitation language detected.")

            if re.search(r"injunctive relief|irreparable harm", text, flags=re.IGNORECASE):
                clause_score += 10
                reasons.append("Strict remedy / injunctive relief language detected.")

            if re.search(r"all information|any information|without limitation", text, flags=re.IGNORECASE):
                clause_score += 15
                reasons.append("Confidential information definition may be overly broad.")

            if re.search(r"assign", text, flags=re.IGNORECASE) and re.search(
                r"without consent", text, flags=re.IGNORECASE
            ):
                clause_score += 10
                reasons.append("Assignment language may be one-sided.")

            if clause_score > 0:
                findings.append(
                    {
                        "page": page,
                        "clause_title": clause_title,
                        "score": clause_score,
                        "reasons": reasons,
                        "supporting_text": text[:500],
                    }
                )
                total_score += clause_score

        overall_score = min(total_score, 100)

        if overall_score >= 70:
            risk_level = "High"
        elif overall_score >= 35:
            risk_level = "Medium"
        else:
            risk_level = "Low"

        return {
            "overall_score": overall_score,
            "risk_level": risk_level,
            "findings": findings,
        }

    # ----------------------------
    # Public features
    # ----------------------------
    def summarize_contract(self) -> str:
        return self.summarize_contract_with_sources()["summary"]

    def summarize_contract_with_sources(self) -> Dict:
        docs = self._retrieve_docs(query=GENERAL_SUMMARY_RETRIEVAL_QUERY, k=8)
        context = self._format_docs(docs)

        fields = self.extract_key_fields_simple()
        field_block = self._format_key_fields_for_summary(fields)
        term_summary = self._get_confidentiality_term_for_summary(fields)

        combined_context = (
            f"{field_block}\n"
            f"Confidentiality Term Helper:\n{term_summary}\n\n"
            f"Retrieved Contract Context:\n{context}"
        )

        prompt = SUMMARY_PROMPT.format(context=combined_context)
        summary = self._call_llm(prompt)

        return {
            "summary": summary,
            "sources": self._docs_to_sources(docs, max_chars=300),
        }

    def analyze_risks(self) -> Dict:
        docs = self._retrieve_docs(
            query=RISK_ANALYSIS_RETRIEVAL_QUERY,
            k=10,
        )
        context = self._format_docs(docs)

        rule_result = self._score_nda_risks(docs)

        rule_summary = f"""
Rule-Based Risk Score: {rule_result['overall_score']}
Rule-Based Risk Level: {rule_result['risk_level']}

Rule Findings:
"""

        for i, item in enumerate(rule_result["findings"], start=1):
            rule_summary += (
                f"{i}. Clause: {item['clause_title']} | Page: {item['page']} | Score: {item['score']}\n"
                f"   Reasons: {'; '.join(item['reasons'])}\n"
            )

        combined_context = f"{rule_summary}\n\nRetrieved Contract Context:\n{context}"

        prompt = RISK_ANALYSIS_PROMPT.format(context=combined_context)
        llm_output = self._call_llm(prompt)

        return {
            "overall_score": rule_result["overall_score"],
            "risk_level": rule_result["risk_level"],
            "findings": rule_result["findings"],
            "llm_analysis": llm_output,
        }

    # ----------------------------
    # Q&A
    # ----------------------------
    def _is_risk_question(self, normalized_question: str) -> bool:
        return any(
            phrase in normalized_question
            for phrase in [
                "risk",
                "risks",
                "fair",
                "unfair",
                "one-sided",
                "one sided",
                "strict",
                "red flag",
                "red flags",
                "concern",
                "concerns",
            ]
        )

    def _answer_general_question(self, question: str) -> str:
        docs = self._retrieve_docs(query=question, k=5)
        context = self._format_docs(docs)
        prompt = QA_PROMPT.format(question=question, context=context)
        return self._call_llm(prompt)

    def answer_question(self, question: str) -> str:
        return self.answer_question_with_sources(question)["answer"]

    def answer_question_with_sources(self, question: str) -> Dict:
        normalized_question = question.lower().strip()

        if self._is_risk_question(normalized_question):
            docs = self._retrieve_docs(query=RISK_QA_RETRIEVAL_QUERY, k=8)
            context = self._format_docs(docs)
            prompt = RISK_QA_PROMPT.format(question=question, context=context)
            answer = self._call_llm(prompt)
            return {
                "answer": answer,
                "sources": self._docs_to_sources(docs),
            }

        docs = self._retrieve_docs(query=question, k=5)
        context = self._format_docs(docs)
        prompt = QA_PROMPT.format(question=question, context=context)
        answer = self._call_llm(prompt)

        return {
            "answer": answer,
            "sources": self._docs_to_sources(docs),
        }