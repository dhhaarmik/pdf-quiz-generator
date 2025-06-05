import openai
import re
from dotenv import load_dotenv
import os
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import simpleSplit
from reportlab.lib import colors
from io import BytesIO
import zipfile

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def extract_text_from_pdfs(uploaded_files):
    import PyPDF2
    full_text = ""
    for uploaded_file in uploaded_files:
        reader = PyPDF2.PdfReader(uploaded_file)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                full_text += page_text + "\n"
    return full_text

def create_vector_store(text):
    return text

def get_relevant_text(db, query):
    return db

def generate_questions(text, q_type, num_questions):
    if q_type == "multiple choice":
        prompt = f"""Create {num_questions} multiple-choice questions with 4 options each from the text below. 
Do not provide answers yet. Format:

Q1: Question?
- Option A
- Option B
- Option C
- Option D

Text:
{text}
"""
    elif q_type == "short answer":
        prompt = f"""Create {num_questions} short answer questions from the text below. Do not provide answers yet.

Format:
Q1: Question 1
Q2: Question 2
...

Text:
{text}
"""
    elif q_type == "long answer":
        prompt = f"""Create {num_questions} long answer questions from the text below. Do not provide answers yet.

Format:
Q1: Question 1
Q2: Question 2
...

Text:
{text}
"""
    else:
        prompt = f"""Create {num_questions} questions from the text below. Do not provide answers yet.

Format:
Q1: Question 1
Q2: Question 2
...

Text:
{text}
"""
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=900,
        temperature=0.7,
    )
    content = response["choices"][0]["message"]["content"].strip()
    if q_type == "multiple choice":
        question_blocks = re.split(r'\n(?=Q\d+:)', content)
        questions = []
        for block in question_blocks:
            lines = block.strip().split('\n')
            if not lines or not lines[0].startswith("Q"):
                continue
            qtext = lines[0]
            opts = lines[1:]
            opts_clean = []
            for opt in opts:
                m = re.match(r'- (.*)', opt)
                if not m:
                    continue
                opt_text = m.group(1)
                opts_clean.append(opt_text)
            questions.append(qtext + '\n' + '\n'.join([f"- {o}" for o in opts_clean]))
        return questions
    else:
        question_blocks = re.findall(r'(Q\d+:.*?)(?=\nQ\d+:|\Z)', content, re.DOTALL)
        questions = [q.strip() for q in question_blocks if q.strip()]
        return questions

def generate_answer(question, context, q_type):
    if q_type == "multiple choice":
        prompt = f"""Provide the correct answer for the following multiple-choice question based only on the text below. Give only the option letter (A, B, C, or D) and the answer text.

Question:
{question}

Text:
{context}
"""
    else:
        prompt = f"""Provide a concise answer for the following question based only on the text below.

Question:
{question}

Text:
{context}
"""
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
        temperature=0.5,
    )
    content = response["choices"][0]["message"]["content"].strip()
    return content

def draw_separator(p, y, width, color=colors.grey, thickness=1, left_pad=20, right_pad=20):
    p.setStrokeColor(color)
    p.setLineWidth(thickness)
    # Add left/right padding so the line does not touch the questions
    p.line(50 + left_pad, y, width - 50 - right_pad, y)

def create_question_paper_pdf(questions, title_info=None):
    buffer = BytesIO()
    p = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    margin = 50
    y = height - 60

    if title_info:
        p.setFillColor(colors.lightblue)
        p.roundRect(margin-10, y-30, width-2*margin+20, 80, 12, stroke=0, fill=1)
        p.setFillColor(colors.black)
        p.setFont("Helvetica-Bold", 16)
        p.drawCentredString(width/2, y+30, title_info.get("university", "").upper())
        p.setFont("Helvetica", 12)
        p.drawCentredString(width/2, y+8, "Examination Question Paper")
        y -= 10
        p.setFont("Helvetica", 10)
        left_info = f"Branch: {title_info.get('branch', '')}"
        right_info = f"Date: {title_info.get('date', '')}"
        p.drawString(margin, y, left_info)
        p.drawRightString(width - margin, y, right_info)
        y -= 16

        left_info2 = f"Subject: {title_info.get('subject', '')}"
        right_info2 = f"Total Marks: {title_info.get('total_marks', '')}   Duration: {title_info.get('duration', '')}"
        p.drawString(margin, y, left_info2)
        p.drawRightString(width - margin, y, right_info2)
        y -= 18

        draw_separator(p, y, width, color=colors.darkblue, thickness=2)
        y -= 18

        p.setFont("Helvetica-Bold", 13)
        p.drawCentredString(width/2, y, "QUESTION PAPER")
        y -= 28
        p.setFont("Helvetica", 11)
    else:
        p.setFont("Helvetica-Bold", 14)
        p.drawString(margin, y, "Generated Questions/Answers")
        y -= 30
        p.setFont("Helvetica", 11)

    section_order = [
        ("multiple choice", "SECTION A: Multiple Choice Questions"),
        ("short answer", "SECTION B: Short Answer Questions"),
        ("long answer", "SECTION C: Long Answer Questions"),
    ]
    section_colors = [colors.teal, colors.orange, colors.darkred]
    for idx, (qtype, section_title) in enumerate(section_order):
        qlist = questions.get(qtype, [])
        if not qlist:
            continue
        # Section Title
        p.setFont("Helvetica-Bold", 12)
        p.setFillColor(section_colors[idx])
        p.drawString(margin, y, section_title)
        y -= 16
        # Thin colored line under section, with padding so it does not touch where the questions start
        draw_separator(p, y+5, width, color=section_colors[idx], thickness=1.5)
        p.setFillColor(colors.black)
        p.setFont("Helvetica", 11)
        y -= 4
        for q in qlist:
            lines = simpleSplit(q, "Helvetica", 11, width-2*margin)
            for line in lines:
                if y < 60:
                    p.showPage()
                    y = height - 60
                    p.setFont("Helvetica-Bold", 13)
                    p.drawCentredString(width/2, y, f"{section_title} (contd.)")
                    y -= 24
                    p.setFont("Helvetica", 11)
                p.drawString(margin+15, y, line)
                y -= 14
            y -= 8
        # End of section separator, thick grey, with extra left/right margin so it doesn't touch the questions
        y -= 2
        draw_separator(p, y, width, color=colors.grey, thickness=2)
        y -= 18

    p.save()
    pdf = buffer.getvalue()
    buffer.close()
    return pdf

def create_answer_sheet_pdf(questions, answers, title_info=None):
    buffer = BytesIO()
    p = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    margin = 50
    y = height - 60

    if title_info:
        p.setFillColor(colors.lightgreen)
        p.roundRect(margin-10, y-30, width-2*margin+20, 80, 12, stroke=0, fill=1)
        p.setFillColor(colors.black)
        p.setFont("Helvetica-Bold", 16)
        p.drawCentredString(width/2, y+30, title_info.get("university", "").upper())
        p.setFont("Helvetica", 12)
        p.drawCentredString(width/2, y+8, "Examination Answer Sheet")
        y -= 10
        p.setFont("Helvetica", 10)
        left_info = f"Branch: {title_info.get('branch', '')}"
        right_info = f"Date: {title_info.get('date', '')}"
        p.drawString(margin, y, left_info)
        p.drawRightString(width - margin, y, right_info)
        y -= 16

        left_info2 = f"Subject: {title_info.get('subject', '')}"
        right_info2 = f"Total Marks: {title_info.get('total_marks', '')}   Duration: {title_info.get('duration', '')}"
        p.drawString(margin, y, left_info2)
        p.drawRightString(width - margin, y, right_info2)
        y -= 18

        draw_separator(p, y, width, color=colors.darkgreen, thickness=2)
        y -= 18

        p.setFont("Helvetica-Bold", 13)
        p.drawCentredString(width/2, y, "ANSWER SHEET")
        y -= 28
        p.setFont("Helvetica", 11)
    else:
        p.setFont("Helvetica-Bold", 14)
        p.drawString(margin, y, "Generated Answers")
        y -= 30
        p.setFont("Helvetica", 11)

    section_order = [
        ("multiple choice", "SECTION A: Answers to Multiple Choice Questions", colors.teal),
        ("short answer", "SECTION B: Answers to Short Answer Questions", colors.orange),
        ("long answer", "SECTION C: Answers to Long Answer Questions", colors.darkred),
    ]
    for qtype, section_title, color in section_order:
        qlist = questions.get(qtype, [])
        alist = answers.get(qtype, [])
        if not qlist or not alist:
            continue
        p.setFont("Helvetica-Bold", 12)
        p.setFillColor(color)
        p.drawString(margin, y, section_title)
        y -= 16
        draw_separator(p, y+5, width, color=color, thickness=1.5)
        p.setFillColor(colors.black)
        p.setFont("Helvetica", 11)
        y -= 4
        for idx, (q, a) in enumerate(zip(qlist, alist), 1):
            q_number_match = re.match(r'(Q\d+:)', q)
            q_number = q_number_match.group(1) if q_number_match else f"Q{idx}:"
            ans_line = f"{q_number} {a}"
            lines = simpleSplit(ans_line, "Helvetica", 11, width-2*margin)
            for line in lines:
                if y < 60:
                    p.showPage()
                    y = height - 60
                    p.setFont("Helvetica-Bold", 13)
                    p.drawCentredString(width/2, y, f"{section_title} (contd.)")
                    y -= 24
                    p.setFont("Helvetica", 11)
                p.drawString(margin+15, y, line)
                y -= 14
            y -= 8
        y -= 2
        draw_separator(p, y, width, color=colors.grey, thickness=2)
        y -= 18

    p.save()
    pdf = buffer.getvalue()
    buffer.close()
    return pdf

def create_zip_file(q_pdf, a_pdf):
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
        zip_file.writestr("question_paper.pdf", q_pdf)
        if a_pdf:
            zip_file.writestr("answer_sheet.pdf", a_pdf)
    zip_buffer.seek(0)
    return zip_buffer.getvalue()
