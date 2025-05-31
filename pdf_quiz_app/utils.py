import fitz  # PyMuPDF
import os
import openai
import faiss
import numpy as np
from dotenv import load_dotenv
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import io

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


def extract_text_from_pdf(file) -> str:
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    file.seek(0)
    return text


def chunk_text(text, max_words=200):
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i : i + max_words])
        chunks.append(chunk)
    return chunks


def embed_texts(texts):
    response = openai.Embedding.create(
        input=texts,
        model="text-embedding-ada-002",
    )
    embeddings = [np.array(data["embedding"], dtype=np.float32) for data in response["data"]]
    return np.vstack(embeddings)


def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index


def retrieve_chunks(index, query, chunks, top_k=5):
    query_embedding = embed_texts([query])
    distances, indices = index.search(query_embedding, top_k)
    results = []
    for idx in indices[0]:
        results.append(chunks[idx])
    return results


def generate_questions(text, num_questions=10, question_type="MCQ"):
    # We'll generate questions from the text passed (already retrieved)
    batch_size = 10
    questions = []

    def generate_batch(batch_num, input_text):
        if question_type == "MCQ":
            prompt = f"""
You are a quiz generator. Create exactly {batch_num} multiple-choice questions with 4 options each from the text below. 
Mark the correct answer with '*'. Format:

Q1: Question?
- Option A
- Option B
- Option C
- Option D

Text:
{input_text}
"""
        else:
            prompt = f"""
You are a quiz generator. Create exactly {batch_num} {question_type.lower()} answer questions from the text below.

Format:
Q1: Question 1
Q2: Question 2
...

Text:
{input_text}
"""

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=800,
        )
        return response['choices'][0]['message']['content']

    total_batches = (num_questions + batch_size - 1) // batch_size
    generated = 0

    for _ in range(total_batches):
        batch_num = min(batch_size, num_questions - generated)
        raw_output = generate_batch(batch_num, text)

        if question_type == "MCQ":
            blocks = [b for b in raw_output.split("\n\n") if b.strip()]
            for block in blocks:
                lines = block.strip().split("\n")
                if len(lines) >= 5:
                    q = lines[0].split(":", 1)[-1].strip()
                    options = []
                    answer = None
                    for opt in lines[1:]:
                        clean_opt = opt.replace("*", "").strip()
                        options.append(clean_opt)
                        if "*" in opt:
                            answer = clean_opt
                    questions.append({"question": q, "options": options, "answer": answer})
        else:
            import re

            matches = re.findall(r"Q\d+:\s*(.+)", raw_output)
            for q in matches:
                questions.append({"question": q.strip(), "options": [], "answer": None})

            # Generate sample answers for short/long questions
            for q in questions[-batch_num:]:
                answer_prompt = f"""
Provide a model answer for the following {question_type.lower()} answer question:

Question: {q['question']}

Base your answer only on the content below:

{text}
"""
                ans_response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": answer_prompt}],
                    temperature=0.7,
                    max_tokens=500,
                )
                q['answer'] = ans_response['choices'][0]['message']['content'].strip()

        generated += batch_num

    return questions[:num_questions]


def generate_question_pdf(questions):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    y = height - 50
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y, "Generated Questions")
    y -= 40
    c.setFont("Helvetica", 12)

    for i, q in enumerate(questions, 1):
        c.drawString(50, y, f"Q{i}: {q['question']}")
        y -= 20
        if q['options']:
            for opt in q['options']:
                c.drawString(70, y, f"- {opt}")
                y -= 20
        else:
            y -= 10
        if y < 100:
            c.showPage()
            y = height - 50

    c.save()
    buffer.seek(0)
    return buffer.read()


def generate_answer_pdf(questions):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    y = height - 50
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y, "Answers")
    y -= 40
    c.setFont("Helvetica", 12)

    for i, q in enumerate(questions, 1):
        c.drawString(50, y, f"Q{i}: {q['question']}")
        y -= 20
        c.drawString(70, y, f"Answer: {q['answer'] if q['answer'] else 'N/A'}")
        y -= 40
        if y < 100:
            c.showPage()
            y = height - 50

    c.save()
    buffer.seek(0)
    return buffer.read()
