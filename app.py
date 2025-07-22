from flask import Flask, request, render_template
import os
import fitz  
from docx import Document
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from google import genai

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

genai_client = genai.Client(api_key="API_KEY")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')



DOCUMENT_TEXT = ""
FAISS_INDEX = None
CHUNKS = []

def extract_text_from_pdf(file_path):
    text = ""
    with fitz.open(file_path) as pdf:
        for page in pdf:
            text += page.get_text()
    return text

def extract_text_from_docx(file_path):
    doc = Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])

def chunk_text(text, chunk_size=100, overlap=50):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

@app.route("/", methods=["GET", "POST"])
def index():
    global DOCUMENT_TEXT, FAISS_INDEX, CHUNKS

    message = ""
    answer = ""

    if request.method == "POST":
        if "file" in request.files:
            file = request.files["file"]
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)

            # Extract text
            if file.filename.endswith('.pdf'):
                text = extract_text_from_pdf(filepath)
            elif file.filename.endswith('.docx'):
                text = extract_text_from_docx(filepath)
            else:
                message = "Unsupported file format. Please upload PDF or DOCX."
                return render_template("index.html", message=message, answer=answer)

            DOCUMENT_TEXT = text

         
            chunks = chunk_text(DOCUMENT_TEXT)
            CHUNKS = chunks

            embeddings = embedding_model.encode(chunks)
            embeddings_np = np.array(embeddings).astype('float32')

            dimension = embeddings_np.shape[1]
            index = faiss.IndexFlatL2(dimension)
            index.add(embeddings_np)

            FAISS_INDEX = index

            message = " File processed, text extracted, and embeddings indexed successfully."

    return render_template("index.html", message=message, answer=answer)


@app.route("/ask", methods=["POST"])
def ask():
    global FAISS_INDEX, CHUNKS

    message = ""
    answer = ""

    question = request.form.get("question", "")

    if not question:
        message = " Please enter a question."
    elif FAISS_INDEX is None:
        message = " No document uploaded yet. Please upload a file first."
    else:
        # Embed question
        q_embedding = embedding_model.encode([question]).astype('float32')
        D, I = FAISS_INDEX.search(q_embedding, 3)
        top_chunks = [CHUNKS[i] for i in I[0]]

        prompt = f"Answer the question based on the context below.\n\nContext:\n{''.join(top_chunks)}\n\nQuestion: {question}\nAnswer:"

        response = genai_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        answer = response.text
        message = " Answer generated successfully."

    return render_template("index.html", message=message, answer=answer)


if __name__ == "__main__":
    app.run(debug=True)
