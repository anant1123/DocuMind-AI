import fitz  # PyMuPDF
import pdfplumber
import pandas as pd
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import logging

# Set up logging
logger = logging.getLogger(__name__)

def extract_text_and_tables_from_pdf(file_path: str) -> str:
    full_text_content = ""
    try:
        with fitz.open(file_path) as doc:
            for page_num, page in enumerate(doc):
                full_text_content += page.get_text("text") + "\n"
                table_areas = page.find_tables()
                if table_areas.tables:
                    logger.info(f"Found {len(table_areas.tables)} table(s) on page {page_num + 1}")
                    with pdfplumber.open(file_path) as pdf:
                        plumber_page = pdf.pages[page_num]
                        tables = plumber_page.extract_tables()
                        for table in tables:
                            if table:
                                df = pd.DataFrame(table[1:], columns=table)
                                full_text_content += "\n" + df.to_markdown(index=False) + "\n"
    except Exception as e:
        logger.error(f"Error processing PDF file {file_path}: {e}", exc_info=True)
        raise ValueError("Failed to process PDF file. It may be corrupted or in an unsupported format.")
        
    return full_text_content

def split_text_into_chunks(text: str, file_path: str) -> List:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.create_documents(
        [text], 
        metadatas=[{"source": file_path}]
    )
    logger.info(f"Split text from {file_path} into {len(chunks)} chunks.")
    return chunks