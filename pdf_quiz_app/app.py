import streamlit as st
from utils import (
    extract_text_from_pdf,
    chunk_text,
    embed_texts,
    build_faiss_index,
    retrieve_chunks,
    generate_questions,
    generate_question_pdf,
    generate_answer_pdf,
)
import numpy as np

st.set_page_config(page_title="PDF to Quiz Generator", layout="wide", page_icon="📄")

st.title("📄 AI PDF to Quiz Generator with FAISS")

st.sidebar.header("📤 Upload & Settings")

uploaded_files = st.sidebar.file_uploader(
    "Upload one or more PDFs", type=["pdf"], accept_multiple_files=True
)
num_questions = st.sidebar.slider("Number of Questions", 1, 20, 10)
question_type = st.sidebar.radio("Question Type", ["MCQ", "Short", "Long"])
generate_button = st.sidebar.button("🚀 Generate Quiz")

if uploaded_files and generate_button:
    with st.spinner("Extracting text from PDFs..."):
        full_text = ""
        for pdf_file in uploaded_files:
            full_text += extract_text_from_pdf(pdf_file) + "\n"

    with st.spinner("Chunking text and creating embeddings..."):
        chunks = chunk_text(full_text, max_words=200)
        embeddings = embed_texts(chunks)
        index = build_faiss_index(embeddings)

    with st.spinner("Retrieving relevant chunks and generating questions..."):
        # For quiz generation, no specific query; just take top chunks
        # (or all chunks if small) — here we just join top 5 chunks for prompt
        relevant_chunks = chunks[:5] if len(chunks) > 5 else chunks
        context = "\n\n".join(relevant_chunks)

        questions = generate_questions(context, num_questions=num_questions, question_type=question_type)

    st.subheader("📘 Generated Questions")

    st.markdown(
        """
    <style>
    .question-box {
        background-color: #e6f2ff;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    for i, q in enumerate(questions, 1):
        content = f"<div class='question-box'><b>Q{i}: {q['question']}</b><br>"
        if q["options"]:
            content += "<ul>"
            for opt in q["options"]:
                content += f"<li>{opt}</li>"
            content += "</ul>"
        if q["answer"] and question_type != "MCQ":
            content += f"<b>Answer:</b> {q['answer']}"
        content += "</div>"
        st.markdown(content, unsafe_allow_html=True)

    question_pdf = generate_question_pdf(questions)
    answer_pdf = generate_answer_pdf(questions)

    st.subheader("📥 Download Options")
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="📄 Download Questions PDF",
            data=question_pdf,
            file_name="questions.pdf",
            mime="application/pdf",
        )
    with col2:
        st.download_button(
            label="✅ Download Answers PDF",
            data=answer_pdf,
            file_name="answers.pdf",
            mime="application/pdf",
        )

else:
    st.info("📎 Upload PDFs and click 'Generate Quiz' to start.")
