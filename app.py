from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import os

# Initialize Flask app
app = Flask(__name__)

# Path to your model
model_path = "E:/Desktop/Project/Brevity_TextSummarization/brevity_model"  # Update this path

# Load the Seq2Seq model and tokenizer
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx'}

# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to generate summary
def generate_summary(text):
    inputs = tokenizer.encode(text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(
        inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Route to handle form submission
@app.route('/', methods=['GET', 'POST'])
def index():
    summary = ""
    
    if request.method == 'POST':
        # Process text input
        text_input = request.form.get('text')
        file_input = request.files.get('file')

        # If text input is provided
        if text_input.strip():
            summary = generate_summary(text_input)
        
        # If a file is uploaded
        elif file_input and allowed_file(file_input.filename):
            filename = secure_filename(file_input.filename)
            file_path = os.path.join('uploads', filename)
            file_input.save(file_path)

            # Extract text from file (Handle .txt, .pdf, .docx)
            file_text = ""
            if filename.endswith('.txt'):
                with open(file_path, 'r', encoding='utf-8') as file:
                    file_text = file.read()
            elif filename.endswith('.pdf'):
                from PyPDF2 import PdfReader
                with open(file_path, 'rb') as file:
                    reader = PdfReader(file)
                    file_text = "".join([page.extract_text() for page in reader.pages])
            elif filename.endswith('.docx'):
                from docx import Document
                doc = Document(file_path)
                file_text = "\n".join([para.text for para in doc.paragraphs])

            if file_text:
                summary = generate_summary(file_text)
            else:
                summary = "Unable to extract text from the uploaded file."
        
        else:
            summary = "Please provide either text or a valid file to summarize (only .txt, .pdf, or .docx allowed)."

    return render_template("index.html", summary=summary)

if __name__ == '__main__':
    app.run(debug=True)
