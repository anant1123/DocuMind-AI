import os
import tempfile
import traceback
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np
import openai

# Load embedding model
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Store PDF chunks and embeddings in memory
pdf_chunks = []
pdf_embeddings = []

openai.api_key = os.getenv("OPENAI_API_KEY")

def extract_text_from_pdf(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        raise RuntimeError(f"Failed to extract text: {str(e)}")

def chunk_text(text, chunk_size=500):
    words = text.split()
    return [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def embed_chunks(chunks):
    try:
        embeddings = embedder.encode(chunks)
        return embeddings
    except Exception as e:
        raise RuntimeError(f"Embedding failed: {str(e)}")

def retrieve_relevant_chunks(question, top_k=3):
    try:
        question_embedding = embedder.encode([question])[0]
        similarities = np.dot(pdf_embeddings, question_embedding)
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [pdf_chunks[i] for i in top_indices]
    except Exception as e:
        raise RuntimeError(f"Retrieval failed: {str(e)}")

def generate_answer(question, context):
    try:
        prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300
        )
        return response.choices[0].message['content'].strip()
    except Exception as e:
        raise RuntimeError(f"Generation failed: {str(e)}")

# Example usage for testing
if __name__ == "__main__":
    try:
        pdf_path = "sample.pdf"  # Change to your PDF file
        text = extract_text_from_pdf(pdf_path)
        chunks = chunk_text(text)
        embeddings = embed_chunks(chunks)
        pdf_chunks = chunks
        pdf_embeddings = embeddings
        question = "Summarize the document in 200 words."
        relevant_chunks = retrieve_relevant_chunks(question)
        context = "\n\n".join(relevant_chunks)
        answer = generate_answer(question, context)
        print("Answer:", answer)
    except Exception as e:
        print("Error:", e)
        traceback.print_exc()
