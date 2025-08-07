import os
import logging
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv

from flask import (Flask, request, render_template, redirect, url_for, flash, session, send_from_directory)
from werkzeug.utils import secure_filename

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

from markdown_it import MarkdownIt
from utils import extract_text_and_tables_from_pdf

# --- Global Constants ---
MODEL_CACHE_DIR = "/app/.cache"
UPLOAD_FOLDER = 'uploads'
VECTOR_STORE_FOLDER = 'vector_stores'
ALLOWED_EXTENSIONS = {'pdf'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024
MAX_PDF_COUNT = 5

# --- App Initialization ---
load_dotenv()
app = Flask(__name__)

# --- Server-Side Session Storage ---
# We store session data here instead of in the client-side cookie
server_side_sessions = {}

# --- Markdown Filter ---
md = MarkdownIt()
@app.template_filter('markdown')
def markdown_filter(text):
    return md.render(text) if text else ""

# --- App Configuration ---
app.config.update(
    UPLOAD_FOLDER=UPLOAD_FOLDER,
    VECTOR_STORE_FOLDER=VECTOR_STORE_FOLDER,
    MAX_CONTENT_LENGTH=MAX_CONTENT_LENGTH,
    SECRET_KEY=os.urandom(24) # SECRET_KEY is still needed for signing the session_id cookie
)

# --- Logging Setup ---
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

# --- Helper Functions ---
def get_google_api_key():
    try:
        key = os.environ.get('GOOGLE_API_KEY')
        if not key:
            raise ValueError("GOOGLE_API_KEY not found in your .env file.")
        return key
    except Exception as e:
        app.logger.error(f"Could not read Google API key from environment: {e}")
        return None

def get_embeddings_model():
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

    if not all_texts:
        return None, None

    embeddings = get_embeddings_model()
    vector_store = FAISS.from_texts(texts=all_texts, embedding=embeddings, metadatas=all_metadatas)
    vector_store_path = os.path.join(app.config['VECTOR_STORE_FOLDER'], f"{session_id}.faiss")
    vector_store.save_local(vector_store_path)
    return vector_store_path, filenames

def run_query(query, vector_store_path, chat_history):
    embeddings = get_embeddings_model()
    vector_store = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
    retriever = vector_store.as_retriever(search_kwargs={"k": 20})

    google_api_key = get_google_api_key()
    if not google_api_key:
        raise ValueError("Google API Key not found.")

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-latest",
        google_api_key=google_api_key,
        temperature=0.5,
        top_p=0.85,
        top_k=45,
        max_output_tokens=1500,
        # FIX: Removed deprecated 'convert_system_message_to_human=True'
    )

    contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    qa_system_prompt = """You are an expert document analysis assistant. Your task is to provide a detailed and accurate answer based *only* on the following context.
If the information is not in the context, say "The information is not available in the provided documents." Do not make up information.
Be concise and clear in your response.

Context:
{context}"""
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    Youtube_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, Youtube_chain)

    return rag_chain.invoke({"chat_history": chat_history, "input": query})

# --- Flask Routes ---

@app.route('/', methods=['GET', 'POST'])
def index():
    # Ensure a session_id exists for server-side storage
    if 'session_id' not in session:
        session['session_id'] = os.urandom(24).hex()
        # Initialize an empty dictionary for new sessions
        if session['session_id'] not in server_side_sessions:
            server_side_sessions[session['session_id']] = {}

    session_id = session['session_id']
    # Get the current user's data from our server-side dictionary
    user_session_data = server_side_sessions.get(session_id, {})

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

                # New upload clears the specific user's server-side session
                vector_store_path, filenames = create_vector_store(files_to_process, session_id)
                if vector_store_path:
                    user_session_data = {
                        'vector_store_path': vector_store_path,
                        'pdf_filenames': filenames,
                        'chat_history': []
                    }
                else:
                    flash('Could not extract text from the uploaded PDFs.', 'error')
                    return redirect(url_for('index'))

            elif 'vector_store_path' not in user_session_data:
                flash('Please attach PDF files with your first question.', 'error')
                return redirect(url_for('index'))

            chat_history_from_session = user_session_data.get('chat_history', [])
            formatted_history = []
            for msg in chat_history_from_session:
                if msg["role"] == "user":
                    formatted_history.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    formatted_history.append(AIMessage(content=msg["content"]))

            result = run_query(query, user_session_data['vector_store_path'], formatted_history)

            user_session_data['chat_history'].append({"role": "user", "content": query})
            user_session_data['chat_history'].append({"role": "assistant", "content": result['answer']})
            user_session_data['sources'] = [doc.metadata.get('source', 'Unknown') + ": " + doc.page_content for doc in result['context']]

            # Save the updated data back to the server-side dictionary
            server_side_sessions[session_id] = user_session_data

        except Exception as e:
            app.logger.error(f"An error occurred: {e}", exc_info=True)
            flash(f"An error occurred: {e}", 'error')

        return redirect(url_for('index'))
    
    # THE FIX IS HERE: Pass the 'session_data' dictionary to the template on GET requests
    return render_template('index.html', session_data=user_session_data)
@app.route('/new_chat')
def new_chat():
    # Clear the server-side data for this session_id
    if 'session_id' in session and session['session_id'] in server_side_sessions:
        # We can just remove the vector store path to prompt for a new upload
        del server_side_sessions[session['session_id']]
    
    # Or to be more thorough, clear the cookie session as well
    session.clear()
    
    flash('New chat session started. Please upload new documents.', 'info')
    return redirect(url_for('index'))

@app.route('/view_pdf/<path:filename>')
def view_pdf_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)