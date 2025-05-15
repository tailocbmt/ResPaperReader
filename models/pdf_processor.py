import os
import logging
import re
from datetime import datetime
from typing import Dict, List, Optional

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

# Keep pdfplumber for backward compatibility
import pdfplumber


class PDFProcessor:
    def __init__(self, upload_dir="../data/uploads"):
        """Initialize PDF processor with upload directory."""
        self.upload_dir = upload_dir
        os.makedirs(self.upload_dir, exist_ok=True)

    def save_pdf(self, pdf_file):
        """
        Save an uploaded PDF file.

        Args:
            pdf_file: File object from Streamlit

        Returns:
            Path to the saved file
        """
        try:
            # Generate a unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_{pdf_file.name}"
            file_path = os.path.join(self.upload_dir, filename)

            # Save the file
            with open(file_path, "wb") as f:
                f.write(pdf_file.getbuffer())

            return file_path
        except Exception as e:
            logging.error(f"Error saving PDF: {e}")
            return None

    def extract_metadata(self, file_path: str) -> Dict:
        """
        Extract title, abstract, and authors from a PDF file using LangChain.

        Args:
            file_path: Path to the PDF file

        Returns:
            Dict containing title, abstract, and authors
        """
        try:
            # Use LangChain's PyPDFLoader
            loader = PyPDFLoader(file_path)
            documents = loader.load()

            # Get text from the first 2 pages which typically contain metadata
            text = ""
            for doc in documents[:2]:
                text += doc.page_content + "\n"

            # Extract title (assume first line with significant text)
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            title = lines[0] if lines else "Unknown Title"

            # Look for abstract section
            abstract = ""
            abstract_match = re.search(
                r'(?:Abstract|ABSTRACT)[\s\.\:\-]*([^\n]+(?:\n[^\n]+)*?)(?:\n\n|\n[A-Z][a-z])', text, re.IGNORECASE)
            if abstract_match:
                abstract = abstract_match.group(1).strip()

            # Extract author information
            authors = []
            author_section = re.search(
                r'(?:^|\n)([^,\n]+(?:,\s*[^,\n]+)*)\n(?:[^@\n]*@[^@\n]*|[Aa]bstract)', text)
            if author_section:
                author_text = author_section.group(1).strip()
                authors = [a.strip() for a in author_text.split(',')]

            return {
                "title": title,
                "abstract": abstract,
                "authors": authors
            }
        except Exception as e:
            logging.error(f"Error extracting metadata from PDF: {e}")
            return {
                "title": os.path.basename(file_path),
                "abstract": "",
                "authors": []
            }

    def extract_full_text(self, file_path: str) -> str:
        """
        Extract full text from a PDF using LangChain.

        Args:
            file_path: Path to the PDF file

        Returns:
            Full text content of the PDF
        """
        try:
            # Use LangChain's PyPDFLoader
            loader = PyPDFLoader(file_path)
            documents = loader.load()

            # Combine text from all documents/pages
            text = "\n\n".join([doc.page_content for doc in documents])
            return text
        except Exception as e:
            logging.error(f"Error extracting full text: {e}")
            return ""

    def extract_documents(self, file_path: str) -> List[Document]:
        """
        Extract pages as LangChain Documents from a PDF.

        Args:
            file_path: Path to the PDF file

        Returns:
            List of LangChain Document objects
        """
        try:
            loader = PyPDFLoader(file_path)
            return loader.load()
        except Exception as e:
            logging.error(
                f"Error loading document as LangChain Documents: {e}")
            return []
