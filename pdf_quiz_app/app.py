import streamlit as st
import openai
import os
from dotenv import load_dotenv
from utils import (
    extract_text_from_pdfs,
    create_vector_store,
    get_relevant_text,
    generate_questions,
    create_pdf,
    create_zip_file
)

# Load API key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Streamlit UI config
st.set_page_config(page_title="📄 PDF to Quiz Generator", layout="wide")
st.title("📘 AI PDF to Quiz Generator")

# Sidebar - Input
with st.sidebar:
    st.header("📂 Upload PDFs")
    uploaded_files = st.file_uploader("Choose one or more PDF files", type="pdf", accept_multiple_files=True)

    st.header("⚙️ Quiz Settings")
    q_type = st.selectbox("Type of Questions", ["multiple choice", "short answer", "long answer"])
    num_questions = st.slider("Number of Questions", 1, 20, 10)
    generate_btn = st.button("🚀 Generate Quiz")

# Main
if generate_btn and uploaded_files:
    with st.spinner("🔍 Reading PDFs..."):
        full_text = extract_text_from_pdfs(uploaded_files)

    with st.spinner("⚙️ Creating vector DB..."):
        db = create_vector_store(full_text)

    with st.spinner("🧠 Generating questions..."):
        query = f"Generate {num_questions} {q_type} questions"
        relevant_text = get_relevant_text(db, query)
        questions, answers = generate_questions(relevant_text, q_type, num_questions)

    st.subheader("📋 Questions")
    st.text(questions)

    st.subheader("✅ Answers")
    st.text(answers)

    q_pdf = create_pdf(questions)
    a_pdf = create_pdf(answers)
    zip_data = create_zip_file(q_pdf, a_pdf)

    st.download_button(
        label="📦 Download ZIP (Questions + Answers)",
        data=zip_data,
        file_name="quiz_bundle.zip",
        mime="application/zip"
    )

elif generate_btn:
    st.warning("⚠️ Please upload at least one PDF file.")
