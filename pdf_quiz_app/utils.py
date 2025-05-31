import fitz  # PyMuPDF
from PyPDF2 import PdfWriter
from io import BytesIO
import openai
import faiss
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
import zipfile
import re


def extract_text_from_pdfs(uploaded_files) -> str:
    full_text = ""
    for file in uploaded_files:
        pdf = fitz.open(stream=file.read(), filetype="pdf")
        for page in pdf:
            full_text += page.get_text()
    return full_text


def create_vector_store(text: str):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_text(text)
    documents = [Document(page_content=t) for t in texts]
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(documents, embeddings)
    return db


def get_relevant_text(db, query: str, k: int = 5) -> str:
    docs = db.similarity_search(query, k=k)
    return "\n".join([doc.page_content for doc in docs])


def format_mcq_vertical(questions: str) -> str:
    pattern = r"([A-D]\))\s?"
    formatted = re.sub(pattern, r"\n\1 ", questions)
    return formatted


def generate_questions(relevant_text: str, q_type: str, num_questions: int) -> (str, str):
    prompt = f"""
You are an expert educator. Based on the following text, generate {num_questions} {q_type} questions and their answers.
Text: {relevant_text}
Return the output in two sections:
1. Questions:
2. Answers:
    """
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    content = response.choices[0].message.content
    if "Answers:" in content:
        parts = content.split("Answers:")
        questions = parts[0].replace("Questions:", "").strip()
        answers = parts[1].strip()
    else:
        questions = content.strip()
        answers = "Not found."

    if q_type == "multiple choice":
        questions = format_mcq_vertical(questions)

    return questions, answers


def create_pdf(text: str) -> BytesIO:
    buffer = BytesIO()
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.units import inch

    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    y = height - inch

    lines = text.split('\n')
    for line in lines:
        if y < inch:
            c.showPage()
            y = height - inch
        c.drawString(inch, y, line)
        y -= 14

    c.save()
    buffer.seek(0)
    return buffer


def create_zip_file(question_pdf: BytesIO, answer_pdf: BytesIO) -> BytesIO:
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED) as zip_file:
        zip_file.writestr("questions.pdf", question_pdf.getvalue())
        zip_file.writestr("answers.pdf", answer_pdf.getvalue())
    zip_buffer.seek(0)
    return zip_buffer
