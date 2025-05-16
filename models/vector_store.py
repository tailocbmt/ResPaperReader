import os
import logging
from typing import List, Dict, Any, Optional

# LangChain imports
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

# Keep compatibility with existing code
import numpy as np
import faiss
import pickle


class VectorStore:
    def __init__(self, persist_directory=None, embedding_model_name="all-MiniLM-L6-v2"):
        """Initialize the LangChain-based vector store."""
        if persist_directory is None:
            # Use absolute path for consistency
            base_dir = os.path.dirname(
                os.path.dirname(os.path.abspath(__file__)))
            self.persist_directory = os.path.join(
                base_dir, "data", "chroma_db")
        else:
            self.persist_directory = persist_directory

        logging.info(
            f"Vector store directory set to: {self.persist_directory}")

        # Ensure directory exists
        os.makedirs(self.persist_directory, exist_ok=True)

        # For backward compatibility
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.index_path = os.path.join(base_dir, "data", "faiss_index")
        self.metadata_path = os.path.join(base_dir, "data", "metadata.pkl")

        try:
            # Initialize embeddings model
            self.embeddings = HuggingFaceEmbeddings(
                model_name=embedding_model_name)

            # Initialize or load Chroma vector store
            self.db = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
            logging.info(
                f"Initialized Chroma vector store with {len(self.db.get()['ids'])} documents")
        except Exception as e:
            logging.error(f"Error initializing vector store: {e}")
            self.db = None

        # Load legacy data if it exists for migration purposes
        self.legacy_metadata = self._load_legacy_metadata()

    def _load_legacy_metadata(self):
        """Load legacy metadata for migration purposes."""
        if os.path.exists(self.metadata_path):
            try:
                with open(self.metadata_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logging.error(f"Error loading legacy metadata: {e}")
        return {}

    def add_document(self, doc_id: str, text: str, paper_info: Optional[Dict] = None) -> tuple:
        """
        Add a document to the vector store using LangChain.

        Args:
            doc_id: Unique identifier for the document
            text: Text to be embedded (title + abstract)
            paper_info: Associated paper metadata

        Returns:
            Tuple of (success, id)
        """
        try:
            if not self.db:
                raise ValueError("Vector store not initialized")

            # Create LangChain document
            metadata = {"doc_id": doc_id}
            if paper_info:
                metadata.update(paper_info)

            document = Document(
                page_content=text,
                metadata=metadata
            )

            # Add to vector store
            ids = self.db.add_documents([document], ids=[str(doc_id)])
            self.db.persist()  # Persist changes

            return True, ids[0]
        except Exception as e:
            logging.error(f"Error adding document to vector store: {e}")
            return False, None

    def search(self, query: str, k: int = 5) -> List[Dict]:
        """
        Search for similar documents using LangChain.

        Args:
            query: Search query text
            k: Number of results to return

        Returns:
            List of document data and scores
        """
        try:
            if not self.db:
                raise ValueError("Vector store not initialized")

            # Search using LangChain
            results = self.db.similarity_search_with_score(query, k=k)

            # Format results
            formatted_results = []
            for doc, score in results:
                doc_id = doc.metadata.get("doc_id")
                formatted_results.append({
                    "doc_id": doc_id,
                    "paper_info": {k: v for k, v in doc.metadata.items() if k != "doc_id"},
                    "score": float(score)
                })

            return formatted_results
        except Exception as e:
            logging.error(f"Error during search: {e}")
            return []

    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        Add multiple LangChain documents to the vector store.

        Args:
            documents: List of LangChain Document objects

        Returns:
            List of document IDs
        """
        try:
            if not self.db:
                raise ValueError("Vector store not initialized")

            # Generate IDs if not present in metadata
            ids = [str(doc.metadata.get("doc_id", i))
                   for i, doc in enumerate(documents)]

            # Add documents to vector store
            self.db.add_documents(documents, ids=ids)
            self.db.persist()  # Persist changes

            return ids
        except Exception as e:
            logging.error(f"Error adding documents to vector store: {e}")
            return []
