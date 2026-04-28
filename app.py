from __future__ import annotations

import os
import tempfile

import streamlit as st

from contract_rag import ContractRAG


st.set_page_config(page_title="AI NDA Analyzer", layout="wide")

st.title("AI NDA Analyzer")
st.caption("AI-powered NDA analysis with OpenAI, RAG, risk scoring, and source citation")


if "rag" not in st.session_state:
    st.session_state.rag = ContractRAG()

if "document_ready" not in st.session_state:
    st.session_state.document_ready = False

if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = None


def save_uploaded_file(uploaded_file) -> str:
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        return tmp_file.name


uploaded_file = st.file_uploader(
    "Upload an NDA / Confidentiality Agreement PDF",
    type=["pdf"]
)

col1, col2 = st.columns([1, 1])

with col1:
    if uploaded_file is not None:
        st.write(f"Selected file: **{uploaded_file.name}**")

        if st.button("Process Agreement"):
            try:
                st.session_state.rag = ContractRAG()
                file_path = save_uploaded_file(uploaded_file)
                num_chunks = st.session_state.rag.ingest_pdf(file_path)

                st.session_state.document_ready = True
                st.session_state.uploaded_file_name = uploaded_file.name

                st.success(f"Agreement processed successfully. Created {num_chunks} chunks.")
            except Exception as e:
                st.error(f"Error while processing agreement: {e}")

with col2:
    if st.session_state.document_ready:
        st.info(f"Current agreement: {st.session_state.uploaded_file_name}")
    else:
        st.warning("No processed agreement yet.")

st.divider()

if st.session_state.document_ready:
    tab1, tab2, tab3 = st.tabs(["Summary", "Risk Analysis", "Q&A"])

    # ----------------------------
    # TAB 1: SUMMARY
    # ----------------------------
    with tab1:
        st.subheader("Plain-Language Summary")

        if st.button("Generate Summary"):
            try:
                with st.spinner("Generating summary..."):
                    summary_result = st.session_state.rag.summarize_contract_with_sources()

                st.write(summary_result["summary"])

                st.subheader("Supporting Sources")
                for src in summary_result["sources"]:
                    st.info(f"Clause: {src['clause_title']} | Page: {src['page']}")
                    st.code(src["supporting_text"])
                    st.divider()

            except Exception as e:
                st.error(f"Summary error: {e}")

    # ----------------------------
    # TAB 2: RISK ANALYSIS
    # ----------------------------
    with tab2:
        st.subheader("Risk Analysis")

        if st.button("Analyze Risks"):
            try:
                with st.spinner("Analyzing risks..."):
                    risk_result = st.session_state.rag.analyze_risks()

                c1, c2 = st.columns(2)
                with c1:
                    st.metric("Overall Risk Score", risk_result["overall_score"])
                with c2:
                    st.metric("Risk Level", risk_result["risk_level"])

                st.subheader("LLM Risk Explanation")
                st.write(risk_result["llm_analysis"])

                st.subheader("Highlighted Risky Clauses")
                if risk_result["findings"]:
                    for item in risk_result["findings"]:
                        st.error(
                            f"Clause: {item['clause_title']} | Page: {item['page']} | Score: {item['score']}"
                        )
                        st.write("**Reasons:**")
                        for reason in item["reasons"]:
                            st.write(f"- {reason}")

                        st.write("**Supporting Text:**")
                        st.code(item["supporting_text"])
                        st.divider()
                else:
                    st.success("No major risky clauses were detected by the rule-based module.")

            except Exception as e:
                st.error(f"Risk analysis error: {e}")

    # ----------------------------
    # TAB 3: Q&A
    # ----------------------------
    with tab3:
        st.subheader("Ask Questions About the Agreement")

        user_question = st.text_input(
            "Enter your question",
            placeholder="Example: How long does confidentiality last?"
        )

        if st.button("Ask"):
            if not user_question.strip():
                st.warning("Please enter a question.")
            else:
                try:
                    with st.spinner("Retrieving answer..."):
                        qa_result = st.session_state.rag.answer_question_with_sources(user_question)

                    st.write("### Answer")
                    st.write(qa_result["answer"])

                    st.write("### Supporting Sources")
                    for src in qa_result["sources"]:
                        st.info(f"Clause: {src['clause_title']} | Page: {src['page']}")
                        st.code(src["supporting_text"])
                        st.divider()

                except Exception as e:
                    st.error(f"Q&A error: {e}")

else:
    st.info("Upload and process an NDA / confidentiality agreement first.")