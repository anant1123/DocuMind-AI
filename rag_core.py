import os
import logging
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from utils import extract_text_and_tables_from_pdf, split_text_into_chunks

# Set up logging
logger = logging.getLogger(__name__)

class PDFQueryEngine:
    def __init__(self, embedding_model_name="BAAI/bge-small-en-v1.5", llm_model="llama3"):
        logger.info("Initializing PDFQueryEngine...")
        self.embedding_model = self._initialize_embedding_model(embedding_model_name)
        self.llm = Ollama(model=llm_model)
        self.vector_store = None
        self.qa_chain = None

    def _initialize_embedding_model(self, model_name):
        logger.info(f"Loading embedding model: {model_name}")
        model_kwargs = {"device": "cpu"}
        encode_kwargs = {"normalize_embeddings": True}
        return HuggingFaceBgeEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )

    def index_pdf(self, file_path: str, store_path: str = "vector_store"):
        logger.info(f"Starting indexing for PDF: {file_path}")
        if not os.path.exists(file_path):
            logger.error(f"File not found at {file_path}")
            raise FileNotFoundError(f"PDF file not found at {file_path}")

        text_content = extract_text_and_tables_from_pdf(file_path)
        if not text_content.strip():
            logger.warning(f"No text content extracted from {file_path}")
            raise ValueError("No text could be extracted from the PDF.")
            
        chunks = split_text_into_chunks(text_content, file_path)
        
        try:
            logger.info("Creating FAISS vector store...")
            self.vector_store = FAISS.from_documents(chunks, self.embedding_model)
            self.vector_store.save_local(store_path)
            logger.info(f"Vector store created and saved at {store_path}")
        except Exception as e:
            logger.error(f"Error creating vector store: {e}", exc_info=True)
            raise RuntimeError("Failed to create vector store.")

        self._setup_qa_chain()
        logger.info("Indexing complete.")

    def _setup_qa_chain(self):
        logger.info("Setting up QA chain...")
        prompt_template = """
        Use the following pieces of context to answer the question at the end.
        If you don't know the answer based on the provided context, just say that you don't know.
        Do not try to make up an answer. Keep the answer concise and professional.

        Context: {context}
        Question: {question}
        Helpful Answer:"""
        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 4}),
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True,
        )
        logger.info("QA chain is ready.")

    def query(self, question: str) -> dict:
        logger.info(f"Received query: {question}")
        if not self.qa_chain:
            logger.error("Query attempted before QA chain was initialized.")
            return {"error": "QA chain not initialized. Please index a document first."}
        
        try:
            # Check for relevance before calling LLM
            retrieved_docs = self.qa_chain.retriever.invoke(question)
            if not retrieved_docs:
                logger.warning("No relevant documents found for the query.")
                return {
                    'result': "I could not find any relevant information in the document to answer your question.",
                    'source_documents': []
                }

            logger.info("Invoking QA chain...")
            result = self.qa_chain.invoke(question)
            logger.info("Query processed successfully.")
            return result
        except Exception as e:
            logger.error(f"Error during query processing: {e}", exc_info=True)
            raise RuntimeError("An error occurred while processing your query.")