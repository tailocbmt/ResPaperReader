import os
from pathlib import Path
import logging
import re
from datetime import datetime
from typing import Dict, List, Optional
import os.path

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


class PDFProcessor:
    def __init__(
        self,
        upload_dir=None
    ):
        """Initialize PDF processor with upload directory."""
        if upload_dir is None:
            # Use absolute path for consistency
            base_dir = os.path.dirname(
                os.path.dirname(os.path.abspath(__file__)))
            self.upload_dir = os.path.join(base_dir, "data", "uploads")
        else:
            self.upload_dir = upload_dir

        logging.info(f"PDF uploads directory set to: {self.upload_dir}")
        os.makedirs(self.upload_dir, exist_ok=True)

    def save_pdf(
            self,
            uploaded_file: str):
        """Save an uploaded PDF file and return its path."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_filename = f"{timestamp}_{uploaded_file.name}"
            destination_path = os.path.join(self.upload_dir, unique_filename)

            with open(destination_path, "wb") as out_file:
                out_file.write(uploaded_file.getbuffer())

            return destination_path
        except Exception as e:
            logging.error(f"Error saving PDF: {e}")
            return None

    def _extract_abstract(self, text: str) -> str:
        """Internal helper to extract abstract section."""
        abstract_search = re.search(
            r'(?:Abstract|ABSTRACT)[\s\.\:\-]*([^\n]+(?:\n[^\n]+)*?)(?:\n\n|\n[A-Z][a-z])',
            text, re.IGNORECASE)
        return abstract_search.group(1).strip() if abstract_search else ""

    def _extract_authors(self, text: str) -> List[str]:
        """Internal helper to extract author list."""
        author_search = re.search(
            r'(?:^|\n)([^,\n]+(?:,\s*[^,\n]+)*)\n(?:[^@\n]*@[^@\n]*|[Aa]bstract)',
            text)
        if author_search:
            return [a.strip() for a in author_search.group(1).split(',')]
        return []

    def extract_metadata(self, pdf_path: str) -> Dict:
        """Extract title, abstract, and authors from a PDF file."""
        try:
            documents = PyPDFLoader(pdf_path).load()
            page_text_combined = "\n".join(
                doc.page_content for doc in documents[:2])
            lines = [line.strip()
                     for line in page_text_combined.splitlines() if line.strip()]

            title = lines[0] if lines else "Unknown Title"
            abstract_text = self._extract_abstract(page_text_combined)
            author_list = self._extract_authors(page_text_combined)

            return {
                "title": title,
                "abstract": abstract_text,
                "authors": author_list
            }
        except Exception as e:
            logging.error(f"Error extracting metadata from PDF: {e}")
            return {
                "title": os.path.basename(pdf_path),
                "abstract": "",
                "authors": []
            }

    def extract_full_text(self, pdf_path: str) -> str:
        """Extract all text from PDF using LangChain."""
        try:
            docs = PyPDFLoader(pdf_path).load()
            return "\n\n".join(doc.page_content for doc in docs)
        except Exception as e:
            logging.error(f"Error extracting full text: {e}")
            return ""

    def extract_documents(self, pdf_path: str) -> List[Document]:
        """Extract PDF pages as LangChain Document objects."""
        try:
            return PyPDFLoader(pdf_path).load()
        except Exception as e:
            logging.error(f"Error loading PDF documents: {e}")
            return []
