import os
import logging
from logging.handlers import RotatingFileHandler
import json
from dotenv import load_dotenv

from flask import (Flask, request, render_template, redirect, url_for, flash, session, send_from_directory)
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

from markdown_it import MarkdownIt
from utils import extract_text_and_tables_from_pdf

# --- NEW: Define the cache directory for the model ---
# This ensures both the download script and the app use the same folder
MODEL_CACHE_DIR = "/app/.cache"

# --- Configuration ---
UPLOAD_FOLDER = 'uploads'
VECTOR_STORE_FOLDER = 'vector_stores'
ALLOWED_EXTENSIONS = {'pdf'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024
MAX_PDF_COUNT = 5

load_dotenv()
app = Flask(__name__)

md = MarkdownIt()
@app.template_filter('markdown')
def markdown_filter(text):
    return md.render(text) if text else ""

app.config.update(
    UPLOAD_FOLDER=UPLOAD_FOLDER,
    VECTOR_STORE_FOLDER=VECTOR_STORE_FOLDER,
    MAX_CONTENT_LENGTH=MAX_CONTENT_LENGTH,
    SECRET_KEY=os.urandom(24)
)

def setup_logging():
    for folder in ['logs', UPLOAD_FOLDER, VECTOR_STORE_FOLDER]:
        if not os.path.exists(folder):
            os.makedirs(folder)
    
    file_handler = RotatingFileHandler('logs/app.log', maxBytes=10240, backupCount=10)
    file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'))
    file_handler.setLevel(logging.INFO)
    app.logger.addHandler(file_handler)
    app.logger.setLevel(logging.INFO)
    app.logger.info('Application startup')

setup_logging()

def get_groq_api_key():
    try:
        key = os.environ.get('GROQ_API_KEY')
        if not key:
            raise ValueError("GROQ_API_KEY not found in environment variables.")
        return key
    except Exception as e:
        app.logger.error(f"Could not read Groq API key from environment: {e}")
        return None

# --- UPDATED: Tell HuggingFaceEmbeddings where the cached model is ---
def get_embeddings_model():
    """Loads the embeddings model, pointing to the pre-downloaded cache."""
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        cache_folder=MODEL_CACHE_DIR
    )

def create_vector_store(files, session_id):
    all_texts, all_metadatas, filenames = [], [], []
    for file in files:
        filename = secure_filename(file.filename)
        filenames.append(filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        text = extract_text_and_tables_from_pdf(file_path)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
        chunks = text_splitter.split_text(text)
        
        all_texts.extend(chunks)
        all_metadatas.extend([{"source": filename}] * len(chunks))
        
    if not all_texts: return None, None
    
    embeddings = get_embeddings_model()
    vector_store = FAISS.from_texts(texts=all_texts, embedding=embeddings, metadatas=all_metadatas)
    vector_store_path = os.path.join(app.config['VECTOR_STORE_FOLDER'], f"{session_id}.faiss")
    vector_store.save_local(vector_store_path)
    
    return vector_store_path, filenames

def run_query(query, vector_store_path, chat_history):
    embeddings = get_embeddings_model()
    vector_store = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
    retriever = vector_store.as_retriever()

    groq_api_key = get_groq_api_key()
    if not groq_api_key: raise ValueError("Groq API Key not found.")
    
    llm = ChatGroq(temperature=0, groq_api_key=groq_api_key, model_name="llama3-8b-8192")

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    
    return qa_chain.invoke({
        "question": query,
        "chat_history": chat_history
    })

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        query = request.form.get('query')
        if not query:
            flash('Please ask a question.', 'error')
            return redirect(url_for('index'))

        try:
            uploaded_files = request.files.getlist('files')
            files_to_process = [f for f in uploaded_files if f.filename]

            if files_to_process:
                if len(files_to_process) > MAX_PDF_COUNT:
                    flash(f'You can upload a maximum of {MAX_PDF_COUNT} PDFs.', 'error')
                    return redirect(url_for('index'))
                
                session.clear()
                session['session_id'] = os.urandom(24).hex()
                vector_store_path, filenames = create_vector_store(files_to_process, session['session_id'])
                
                if vector_store_path:
                    session['vector_store_path'] = vector_store_path
                    session['pdf_filenames'] = filenames
                    session['chat_history'] = []
                else:
                    flash('Could not extract text from the uploaded PDFs.', 'error')
                    return redirect(url_for('index'))
            
            elif 'vector_store_path' not in session:
                flash('Please attach PDF files with your first question.', 'error')
                return redirect(url_for('index'))

            chat_history_from_session = session.get('chat_history', [])
            formatted_history = []
            for i in range(0, len(chat_history_from_session), 2):
                user_msg = chat_history_from_session[i]['content']
                ai_msg = chat_history_from_session[i+1]['content']
                formatted_history.append((user_msg, ai_msg))

            result = run_query(query, session['vector_store_path'], formatted_history)
            
            current_chat_history = session.get('chat_history', [])
            current_chat_history.append({"role": "user", "content": query})
            current_chat_history.append({"role": "assistant", "content": result['answer']})
            session['chat_history'] = current_chat_history
            session['sources'] = [doc.metadata.get('source', 'Unknown') + ": " + doc.page_content for doc in result['source_documents']]

        except Exception as e:
            app.logger.error(f"An error occurred: {e}", exc_info=True)
            flash(f"An error occurred: {e}", 'error')

        return redirect(url_for('index'))
    return render_template('index.html')

@app.route('/new_chat')
def new_chat():
    session.clear()
    flash('New chat session started.', 'info')
    return redirect(url_for('index'))

@app.route('/view_pdf/<path:filename>')
def view_pdf_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)