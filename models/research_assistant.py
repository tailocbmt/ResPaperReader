import os
import logging
import re
import json
from typing import List, Dict, Any, Tuple

from models.database import DatabaseManager
from models.vector_store import VectorStore
from models.pdf_processor import PDFProcessor
from models.api_service import APIService
from models.gemini_agent import GeminiAgent


class ResearchAssistant:
    def __init__(self, gemini_api_key=None):
        """Initialize the research assistant with all necessary components."""
        self.db = DatabaseManager()
        self.vector_store = VectorStore()
        self.pdf_processor = PDFProcessor()
        self.api_service = APIService()
        self.agent = GeminiAgent(api_key=gemini_api_key)

        # Session memory for conversation context
        self.session_memory = {
            "uploaded_papers": [],
            "search_results": [],
            "last_comparison": None
        }

        # Tool mapping
        self.tools = {
            "internal_search": self.search_internal_papers,
            "web_search": self.search_web_papers,
            "conference_search": self.search_conference_papers,
            "compare_papers": self.generate_paper_comparison
        }

    def upload_paper(self, pdf_file):
        """
        Upload and process a research paper.

        Args:
            pdf_file: File object from Streamlit

        Returns:
            Dict with paper information and success status
        """
        try:
            # Save the file
            file_path = self.pdf_processor.save_pdf(pdf_file)
            if not file_path:
                return {"success": False, "message": "Failed to save PDF file"}

            # Extract metadata
            metadata = self.pdf_processor.extract_metadata(file_path)

            # Store in database
            paper_id = self.db.add_paper(
                title=metadata["title"],
                abstract=metadata["abstract"],
                authors=metadata["authors"],
                source="internal_upload",
                file_path=file_path
            )

            # Create vector embedding
            combined_text = f"{metadata['title']} {metadata['abstract']}"
            success, embedding_id = self.vector_store.add_document(
                doc_id=paper_id,
                text=combined_text,
                paper_info={
                    "title": metadata["title"], "abstract": metadata["abstract"], "id": paper_id}
            )

            # Update database with embedding ID
            if success:
                # Update session memory
                self.session_memory["uploaded_papers"].append({
                    "id": paper_id,
                    "title": metadata["title"],
                    "abstract": metadata["abstract"],
                    "path": file_path
                })

                return {
                    "success": True,
                    "paper_id": paper_id,
                    "title": metadata["title"],
                    "abstract": metadata["abstract"],
                    "authors": metadata["authors"]
                }
            else:
                return {"success": False, "message": "Failed to create embedding"}

        except Exception as e:
            logging.error(f"Error uploading paper: {e}")
            return {"success": False, "message": str(e)}

    def search_internal_papers(self, query: str) -> List[Dict]:
        """
        Search for papers in the internal database.

        Args:
            query: Search query

        Returns:
            List of matching papers
        """
        # Try semantic search first
        semantic_results = self.vector_store.search(query, k=5)

        # If no semantic results, fall back to keyword search
        if not semantic_results:
            keyword_results = self.db.search_papers(query)
            return keyword_results

        # Get full paper details from DB for each semantic result
        results = []
        for item in semantic_results:
            if item.get('doc_id'):
                paper = self.db.get_paper(item['doc_id'])
                if paper:
                    paper['score'] = item.get('score', 0)
                    results.append(paper)

        # Update session memory
        self.session_memory["search_results"] = results
        return results

    def search_web_papers(self, query: str, source: str = None) -> List[Dict]:
        """
        Search for papers using external APIs.

        Args:
            query: Search query
            source: Optional source to search (arxiv or semantic_scholar)

        Returns:
            List of matching papers
        """
        results = []
        if source == "arxiv" or source is None:
            arxiv_results = self.api_service.search_arxiv(query)
            results.extend(arxiv_results)

        if source == "semantic_scholar" or source is None:
            scholar_results = self.api_service.search_semantic_scholar(query)
            results.extend(scholar_results)

        # Update session memory
        self.session_memory["search_results"] = results
        return results

    def search_conference_papers(self, conference: str, year: str = None) -> List[Dict]:
        """
        Search for papers from a specific conference.

        Args:
            conference: Conference name
            year: Optional conference year

        Returns:
            List of matching papers
        """
        results = self.api_service.search_papers_by_conference(
            conference, year)

        # Update session memory
        self.session_memory["search_results"] = results
        return results

    def generate_paper_comparison(self, paper_id_1: str, paper_id_2: str) -> str:
        """
        Compare two papers and generate a report.

        Args:
            paper_id_1: ID or index of first paper
            paper_id_2: ID or index of second paper

        Returns:
            Comparison report
        """
        # Helper to get paper by ID or index from memory
        def get_paper(paper_id):
            try:
                # Try as direct DB ID
                paper = self.db.get_paper(int(paper_id))
                if paper:
                    return paper
            except ValueError:
                # Not a valid integer ID
                pass

            # Try as index in recent results
            try:
                idx = int(paper_id)
                if 0 <= idx < len(self.session_memory["search_results"]):
                    return self.session_memory["search_results"][idx]
            except (ValueError, IndexError):
                # Not a valid index
                pass

            return None

        # Get papers to compare
        paper1 = get_paper(paper_id_1)
        paper2 = get_paper(paper_id_2)

        if not paper1 or not paper2:
            return "Could not find one or both papers to compare. Please provide valid paper IDs or indices."

        # Generate comparison using Gemini
        comparison = self.agent.compare_papers(paper1, paper2)

        # Update session memory
        self.session_memory["last_comparison"] = {
            "papers": [paper1, paper2],
            "report": comparison
        }

        return comparison

    def process_natural_language_query(self, query: str) -> Dict:
        """
        Process a natural language query from the user.

        Args:
            query: User's query

        Returns:
            Response from the assistant
        """
        if query.lower().startswith("upload"):
            return {
                "action": "upload_prompt",
                "message": "Please upload a PDF file to continue."
            }

        # Use the Gemini agent to determine intent and action
        context = {
            "uploaded_papers": [p.get("title") for p in self.session_memory["uploaded_papers"]],
            "recent_searches": [p.get("title") for p in self.session_memory["search_results"]]
        }

        response = self.agent.process_query(query, context)

        # Extract tool calls
        tool_pattern = r"<tool>(\w+)\((.*?)\)</tool>"
        tool_matches = re.findall(tool_pattern, response)

        results = []
        for tool_name, params_str in tool_matches:
            # Parse parameters
            params = {}
            param_pattern = r'(\w+):\s*"([^"]*)"'
            param_matches = re.findall(param_pattern, params_str)

            for param_name, param_value in param_matches:
                params[param_name] = param_value

            # Execute tool if available
            if tool_name in self.tools:
                tool_result = self.tools[tool_name](**params)
                results.append({
                    "tool": tool_name,
                    "result": tool_result
                })

        if results:
            return {
                "action": "tool_results",
                "results": results,
                "message": response.replace(tool_pattern, "")
            }
        else:
            return {
                "action": "response",
                "message": response
            }

    def analyze_paper(self, paper_id: int) -> Dict:
        """
        Analyze a paper using the Gemini agent.

        Args:
            paper_id: ID of the paper to analyze

        Returns:
            Analysis results
        """
        paper = self.db.get_paper(paper_id)
        if not paper:
            return {"error": "Paper not found"}

        return self.agent.analyze_paper(paper["title"], paper["abstract"])
