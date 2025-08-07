
🧠 DocuMind AI: Intelligent Document Chat
DocuMind AI is a powerful and intuitive web application that allows you to chat with your documents. Upload multiple PDFs and ask questions to get intelligent, context-aware answers instantly. This project uses a Retrieval-Augmented Generation (RAG) pipeline, powered by the latest models, to provide accurate information grounded in the content of your uploaded files.

✨ Features
Multi-PDF Chat: Upload and analyze up to 5 PDF documents simultaneously.

High-Speed LLM: Powered by the Groq API and Llama 3 for incredibly fast and accurate responses.

Advanced Text Extraction: Capable of parsing both text and tables from your PDFs to ensure no information is missed.

Efficient Retrieval: Uses FAISS for lightning-fast similarity searches to find the most relevant information in your documents.

Clean UI: A simple and intuitive single-page web interface for a seamless user experience.

Containerized: Fully containerized with Docker and a production-ready Gunicorn server for easy setup and deployment on any machine.

⚙️ How It Works: The RAG Workflow
The application follows a sophisticated Retrieval-Augmented Generation (RAG) process to provide answers grounded in the documents you provide.

PDF Parsing & Text Extraction: When you upload PDFs, the application uses PyMuPDF and pdfplumber to meticulously extract all text and even data from tables within the documents.

Chunking: The extracted text is intelligently broken down into smaller, manageable chunks using LangChain's RecursiveCharacterTextSplitter.

Embedding & Indexing: Each chunk is converted into a numerical representation (an embedding) using the all-MiniLM-L6-v2 model. These embeddings are then indexed in a session-specific FAISS vector store for rapid retrieval.

Querying: When you ask a question, it is also converted into an embedding using the same model.

Retrieval: The application performs an efficient similarity search on the FAISS vector store to find the most relevant chunks of text from your documents that relate to your question.

Generation: The original question and the retrieved text chunks are passed as context to the Groq Llama 3 model, which generates a coherent and contextually accurate answer.

🚀 How to Run with Docker
Running this application on your local machine is simple with Docker.

Prerequisites
You must have Docker installed on your system.

You need a Groq API Key. You can get one for free from GroqCloud.

Steps
Clone the Repository
Open your terminal and clone the project repository:

Bash

git clone <your-repository-url>
cd <repository-directory>
Create an Environment File
Create a file named .env in the project root. The application is configured to load your secret key from this file, which is ignored by Git to keep your key private.

GROQ_API_KEY="YOUR_API_KEY_HERE"
Build the Docker Image
Use the provided Dockerfile to build the application image.

Bash

docker build -t documind-ai .
Run the Docker Container
Run the following command in your terminal. This will start the application using the Gunicorn WSGI server inside the container.

Bash

docker run -p 5000:5000 --env-file .env --name documind-app documind-ai
-p 5000:5000 maps your local port 5000 to the container's exposed port 5000.

--env-file .env securely loads your API key as an environment variable inside the container.

Access the Application
Open your web browser and navigate to: http://localhost:5000
You can now upload your PDFs and start asking questions!

📁 Project Structure
The project is organized as follows:

/
├── app.py              # Main Flask application logic and RAG pipeline
├── Dockerfile          # Instructions to build the Docker image
├── requirements.txt    # Python dependencies for the project
├── download_model.py   # Utility script to pre-download the embedding model
├── utils.py            # Helper functions for PDF text/table extraction
└── templates/
    └── index.html      # The single-page frontend for the user interface
