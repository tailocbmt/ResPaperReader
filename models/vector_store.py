import os
import logging
from typing import List, Dict, Any, Optional

# LangChain imports
import chromadb
from langchain_chroma.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

# Keep compatibility with existing code
import numpy as np
import pickle


class VectorStore:
    def __init__(self, store_dir=None, embedding_model_id="all-MiniLM-L6-v2"):
        """Initialize the LangChain-based vector store."""
        if store_dir is None:
            base_path = os.path.dirname(
                os.path.dirname(os.path.abspath(__file__)))
            self.store_dir = os.path.join(base_path, "data", "chroma_db")
        else:
            self.store_dir = store_dir

        logging.info(f"Vector store directory set to: {self.store_dir}")
        os.makedirs(self.store_dir, exist_ok=True)

        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.faiss_path = os.path.join(base_path, "data", "faiss_index")
        self.legacy_meta_path = os.path.join(base_path, "data", "metadata.pkl")

        try:
            self.embedding_model = HuggingFaceEmbeddings(
                model_name=embedding_model_id)
            persistent_client = chromadb.PersistentClient(path=self.store_dir)

            self.db_client = Chroma(
                client=persistent_client,
                collection_name="persist",
                client_settings=chromadb.config.Settings(is_persistent=True),
                persist_directory=self.store_dir,
                embedding_function=self.embedding_model
            )
            logging.info(
                f"Initialized Chroma store with {len(self.db_client.get()['ids'])} documents.")
        except Exception as e:
            logging.error(f"Error initializing vector store: {e}")
            self.db_client = None

        self.legacy_metadata = self._load_legacy_metadata()

    def _load_legacy_metadata(self):
        """Load legacy metadata for migration."""
        if os.path.exists(self.legacy_meta_path):
            try:
                with open(self.legacy_meta_path, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                logging.error(f"Error loading legacy metadata: {e}")
        return {}

    def add_document(self, document_id: str, document_text: str, metadata_extra: Optional[Dict] = None) -> tuple:
        """Add a single document to the vector store."""
        try:
            if not self.db_client:
                raise ValueError("Vector store not initialized.")

            metadata = {"doc_id": document_id}
            if metadata_extra:
                metadata.update(metadata_extra)

            document = Document(page_content=document_text, metadata=metadata)
            added_ids = self.db_client.add_documents(
                [document], ids=[str(document_id)])

            return True, added_ids[0]
        except Exception as e:
            logging.error(f"Error adding document: {e}")
            return False, None

    def search(
            self,
            query: str,
            top_k: int = 5
    ) -> List[Dict]:
        """
        Search for similar documents using LangChain.

        Args:
            query: Search query text
            k: Number of results to return

        Returns:
            List of document data and scores
        """
        try:
            if not self.db_client:
                raise ValueError("Vector store not initialized")

            # Search using LangChain
            results = self.db_client.similarity_search_with_score(
                query,
                k=top_k
            )

            # Format results
            formatted_results = []
            for doc, score in results:
                doc_id = doc.metadata.get("doc_id")
                formatted_results.append(
                    {
                        "doc_id": doc_id,
                        "paper_info": {k: v for k, v in doc.metadata.items() if k != "doc_id"},
                        "score": float(score)
                    }
                )

            return formatted_results
        except Exception as e:
            logging.error(f"Error during search: {e}")
            return []

    def add_documents(self, document_list: List[Document]) -> List[str]:
        """Add multiple documents to the vector store."""
        try:
            if not self.db_client:
                raise ValueError("Vector store not initialized.")

            ids = [str(doc.metadata.get("doc_id", i))
                   for i, doc in enumerate(document_list)]

            ids = self.db_client.add_documents(document_list)
            return ids
        except Exception as e:
            logging.error(f"Error adding documents: {e}")
            return []

    def search(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for documents similar to the query."""
        try:
            if not self.db_client:
                raise ValueError("Vector store not initialized.")

            raw_results = self.db_client.similarity_search_with_score(
                query_text, k=top_k)

            search_results = []
            for doc, score in raw_results:
                document_id = doc.metadata.get("doc_id")
                filtered_metadata = {
                    k: v for k, v in doc.metadata.items() if k != "doc_id"}
                search_results.append({
                    "doc_id": document_id,
                    "paper_info": filtered_metadata,
                    "score": float(score)
                })

            return search_results
        except Exception as e:
            logging.error(f"Error during search: {e}")
            return []
