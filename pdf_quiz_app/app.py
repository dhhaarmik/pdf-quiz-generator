import streamlit as st
import openai
import os
import base64
from dotenv import load_dotenv
from utils import (
    extract_text_from_pdfs,
    create_vector_store,
    get_relevant_text,
    generate_questions,
    generate_answer,
    create_question_paper_pdf,
    create_answer_sheet_pdf,
    create_zip_file,
)

# PDF preview helper
def show_pdf(pdf_bytes):
    base64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="800" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

# Load API key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="📝 Question Paper Generator", layout="wide")
st.title("📝 AI Question Paper Generator")

with st.sidebar:
    st.header("📂 Upload PDFs")
    uploaded_files = st.file_uploader("Upload the Lesson PDF(s)", type="pdf", accept_multiple_files=True)

    st.header("📝 Question Paper Details")
    university = st.text_input("University Name")
    exam_date = st.date_input("Date")
    branch = st.text_input("Branch")
    subject = st.text_input("Subject")
    total_marks = st.number_input("Total Marks", min_value=0, value=100)
    duration = st.text_input("Duration (e.g., 3 Hours)")

    st.header("⚙️ Question Types & Counts")
    question_types = ["multiple choice", "short answer", "long answer"]
    selected_types = st.multiselect("Question Types", question_types, default=question_types)
    num_questions_dict = {}
    for qtype in selected_types:
        num = st.number_input(
            f"Number of {qtype.title()} Questions", min_value=1, max_value=20, value=5, key=qtype
        )
        num_questions_dict[qtype] = num

    generate_btn = st.button("🚀 Generate Question Paper and Answer Sheet")

if generate_btn:
    if not uploaded_files:
        st.warning("⚠️ Please upload at least one PDF file.")
    elif not university or not branch or not subject or not duration:
        st.warning("⚠️ Please fill in all the question paper details.")
    elif not selected_types:
        st.warning("⚠️ Please select at least one question type.")
    else:
        with st.spinner("🔍 Reading PDFs..."):
            full_text = extract_text_from_pdfs(uploaded_files)

        with st.spinner("⚙️ Creating vector DB..."):
            db = create_vector_store(full_text)

        qp_questions_dict = {}
        qp_answers_dict = {}

        with st.spinner("🧠 Generating questions and answers..."):
            for qtype in selected_types:
                num_questions = num_questions_dict[qtype]
                query = f"Generate {num_questions} {qtype} questions"
                relevant_text = get_relevant_text(db, query)
                questions = generate_questions(relevant_text, qtype, num_questions)
                answers = []
                for q in questions:
                    answer = generate_answer(q, relevant_text, qtype)
                    answers.append(answer)
                qp_questions_dict[qtype] = questions
                qp_answers_dict[qtype] = answers

        title_info = {
            "university": university,
            "date": str(exam_date),
            "branch": branch,
            "subject": subject,
            "total_marks": str(total_marks),
            "duration": duration,
        }

        qp_pdf = create_question_paper_pdf(
            questions=qp_questions_dict,
            title_info=title_info
        )
        answer_pdf = create_answer_sheet_pdf(
            questions=qp_questions_dict,
            answers=qp_answers_dict,
            title_info=title_info
        )
        zip_data = create_zip_file(qp_pdf, answer_pdf)

        st.success("Question paper and answer sheet generated!")

        st.subheader("📥 Downloads")
        st.download_button(
            label="Download Question Paper (PDF)",
            data=qp_pdf,
            file_name="question_paper.pdf",
            mime="application/pdf"
        )
        st.download_button(
            label="Download Answer Sheet (PDF)",
            data=answer_pdf,
            file_name="answer_sheet.pdf",
            mime="application/pdf"
        )
        st.download_button(
            label="Download ZIP (Question Paper + Answer Sheet)",
            data=zip_data,
            file_name="question_paper_bundle.zip",
            mime="application/zip"
        )
        st.markdown("---")
        st.markdown("#### 📄 Preview of Question Paper:")
        show_pdf(qp_pdf)
