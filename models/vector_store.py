import os
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer
import logging


class VectorStore:
    def __init__(self, index_path="../data/faiss_index", metadata_path="../data/metadata.pkl", model_name="all-MiniLM-L6-v2"):
        """Initialize the vector store."""
        self.index_path = index_path
        self.metadata_path = metadata_path

        # Ensure directories exist
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.metadata_path), exist_ok=True)

        # Initialize the embedding model (lightweight for Raspberry Pi)
        try:
            self.model = SentenceTransformer(model_name)
            self.dimension = self.model.get_sentence_embedding_dimension()
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            # Fallback dimension if model fails to load
            self.dimension = 384

        # Initialize or load FAISS index
        self._initialize_index()

        # Initialize or load metadata
        self.metadata = self._load_metadata()

    def _initialize_index(self):
        """Initialize or load a FAISS index."""
        if os.path.exists(self.index_path):
            try:
                self.index = faiss.read_index(self.index_path)
                logging.info(
                    f"Loaded existing index with {self.index.ntotal} entries")
            except Exception as e:
                logging.error(f"Error loading index: {e}")
                self.index = faiss.IndexFlatL2(self.dimension)
        else:
            self.index = faiss.IndexFlatL2(self.dimension)
            logging.info("Created new FAISS index")

    def _load_metadata(self):
        """Load metadata dictionary."""
        if os.path.exists(self.metadata_path):
            try:
                with open(self.metadata_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logging.error(f"Error loading metadata: {e}")
                return {}
        else:
            return {}

    def _save_index(self):
        """Save the FAISS index to disk."""
        try:
            faiss.write_index(self.index, self.index_path)
        except Exception as e:
            logging.error(f"Error saving index: {e}")

    def _save_metadata(self):
        """Save metadata to disk."""
        try:
            with open(self.metadata_path, 'wb') as f:
                pickle.dump(self.metadata, f)
        except Exception as e:
            logging.error(f"Error saving metadata: {e}")

    def add_document(self, doc_id, text, paper_info=None):
        """
        Add a document to the vector store.

        Args:
            doc_id: Unique identifier for the document
            text: Text to be embedded (title + abstract)
            paper_info: Associated paper metadata

        Returns:
            True if successful, False otherwise
        """
        try:
            # Create embedding
            vector = self.model.encode([text])[0]
            vector = np.array([vector]).astype('float32')

            # Add to FAISS index
            self.index.add(vector)
            vector_id = self.index.ntotal - 1

            # Store mapping and metadata
            self.metadata[vector_id] = {
                'doc_id': doc_id,
                'paper_info': paper_info
            }

            # Save changes
            self._save_index()
            self._save_metadata()

            return True, vector_id
        except Exception as e:
            logging.error(f"Error adding document: {e}")
            return False, None

    def search(self, query, k=5):
        """
        Search for similar documents.

        Args:
            query: Search query text
            k: Number of results to return

        Returns:
            List of document IDs and scores
        """
        if self.index.ntotal == 0:
            return []

        try:
            # Encode query
            query_vector = self.model.encode([query])[0]
            query_vector = np.array([query_vector]).astype('float32')

            # Search index
            k = min(k, self.index.ntotal)
            distances, indices = self.index.search(query_vector, k)

            # Format results
            results = []
            for i, idx in enumerate(indices[0]):
                if idx >= 0:  # Valid index
                    doc_data = self.metadata.get(int(idx), {})
                    results.append({
                        'doc_id': doc_data.get('doc_id'),
                        'paper_info': doc_data.get('paper_info'),
                        'score': float(distances[0][i])
                    })

            return results
        except Exception as e:
            logging.error(f"Error during search: {e}")
            return []
