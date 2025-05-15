import os
import logging
import re
import json
from typing import List, Dict, Any, Optional, Tuple

from models.database import DatabaseManager
from models.vector_store import VectorStore
from models.pdf_processor import PDFProcessor
from models.api_service import APIService
from models.gemini_agent import GeminiAgent

# LangChain imports
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.chains import ConversationalRetrievalChain
from langchain_core.documents import Document


class ResearchAssistant:
    def __init__(self, gemini_api_key=None):
        """Initialize the research assistant with all necessary components using LangChain."""
        self.db = DatabaseManager()
        self.vector_store = VectorStore()
        self.pdf_processor = PDFProcessor()
        self.api_service = APIService()
        self.agent = GeminiAgent(api_key=gemini_api_key)
        self.api_key = gemini_api_key or os.environ.get("GEMINI_API_KEY")

        # Initialize the LLM
        try:
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash", temperature=0.2, google_api_key=self.api_key)
        except Exception as e:
            logging.error(f"Error initializing LangChain LLM: {e}")
            self.llm = None

        # Session memory for conversation context
        self.session_memory = {
            "uploaded_papers": [],
            "search_results": [],
            "last_comparison": None
        }

        # Initialize LangChain tools and agent
        self._initialize_tools()
        self._initialize_agent()

    def _initialize_tools(self):
        """Initialize LangChain tools."""
        self.tools = [
            Tool(
                name="internal_search",
                func=self.search_internal_papers,
                description="Search for papers in the internal database using keywords or semantic search"
            ),
            Tool(
                name="web_search",
                func=self.search_web_papers,
                description="Search for papers using external APIs like arXiv and Semantic Scholar"
            ),
            Tool(
                name="conference_search",
                func=self.search_conference_papers,
                description="Search for papers from a specific conference"
            ),
            Tool(
                name="compare_papers",
                func=self.generate_paper_comparison,
                description="Compare two research papers and generate a structured report"
            )
        ]

    def _initialize_agent(self):
        """Initialize LangChain agent with tools."""
        if not self.llm:
            self.agent_executor = None
            return

        try:
            # Create system prompt
            system_prompt = """You are a Research Paper Assistant that helps users find, analyze, and compare research papers.
            You can understand natural language commands and use tools to find relevant information.
            Always be concise and focus on academic research needs. Format your responses in a clear, informative way.
            """

            # Create prompt template
            prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content=system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                HumanMessage(content="{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad")
            ])

            # Create agent
            agent = create_react_agent(self.llm, self.tools, prompt)

            # Create agent executor
            self.agent_executor = AgentExecutor(
                agent=agent,
                tools=self.tools,
                verbose=True
            )

            logging.info("Successfully initialized LangChain agent")
        except Exception as e:
            logging.error(f"Error initializing LangChain agent: {e}")
            self.agent_executor = None

    def upload_paper(self, pdf_file):
        """
        Upload and process a research paper with LangChain.

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

            # Extract documents using LangChain
            documents = self.pdf_processor.extract_documents(file_path)

            # Set document metadata
            for doc in documents:
                doc.metadata.update({
                    "doc_id": paper_id,
                    "title": metadata["title"],
                    "abstract": metadata["abstract"]
                })

            # Add to vector store using LangChain documents
            self.vector_store.add_documents(documents)

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

        except Exception as e:
            logging.error(f"Error uploading paper: {e}")
            return {"success": False, "message": str(e)}

    def search_internal_papers(self, query: str) -> List[Dict]:
        """
        Search for papers in the internal database using LangChain.

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
        Compare two papers using LangChain.

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

        # Generate comparison using LangChain
        comparison = self.agent.compare_papers(paper1, paper2)

        # Update session memory
        self.session_memory["last_comparison"] = {
            "papers": [paper1, paper2],
            "report": comparison
        }

        return comparison

    def create_retrieval_chain(self, paper_id: str):
        """
        Create a LangChain retrieval chain for a specific paper.

        Args:
            paper_id: ID of the paper to create a retrieval chain for

        Returns:
            A ConversationalRetrievalChain for the paper
        """
        try:
            paper = self.db.get_paper(paper_id)
            if not paper or not paper.get('file_path'):
                return None

            # Extract documents from the paper
            documents = self.pdf_processor.extract_documents(
                paper['file_path'])

            # Create a retriever
            retriever = self.vector_store.db.as_retriever(
                search_kwargs={"k": 5, "filter": {"doc_id": paper_id}}
            )

            # Create the chain
            chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=retriever,
                return_source_documents=True,
                verbose=True
            )

            return chain
        except Exception as e:
            logging.error(f"Error creating retrieval chain: {e}")
            return None

    def process_natural_language_query(self, query: str) -> Dict:
        """
        Process a natural language query from the user using LangChain agent.

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

        if not self.agent_executor:
            # Fall back to original implementation if LangChain agent isn't available
            return self._legacy_process_query(query)

        try:
            # Use LangChain agent executor to process the query
            chat_history = []
            context = {
                "uploaded_papers": [p.get("title") for p in self.session_memory["uploaded_papers"]],
                "recent_searches": [p.get("title") for p in self.session_memory["search_results"]]
            }

            # Add context to query
            enhanced_query = f"{query}\n\nContext: {json.dumps(context)}"

            # Execute agent
            result = self.agent_executor.invoke(
                {"input": enhanced_query, "chat_history": chat_history}
            )

            response = result["output"]

            # Process results from tool calls
            if "intermediate_steps" in result:
                tools_used = []
                for step in result["intermediate_steps"]:
                    tool_name = step[0].tool
                    tool_result = step[1]

                    if isinstance(tool_result, list) and len(tool_result) > 0:  # Search results
                        tools_used.append({
                            "tool": tool_name,
                            "result": tool_result
                        })

                if tools_used:
                    return {
                        "action": "tool_results",
                        "results": tools_used,
                        "message": response
                    }

            # Regular response (no tools used)
            return {
                "action": "response",
                "message": response
            }

        except Exception as e:
            logging.error(f"Error in LangChain agent: {e}")
            # Fall back to original implementation
            return self._legacy_process_query(query)

    def _legacy_process_query(self, query: str) -> Dict:
        """Legacy query processing when LangChain agent isn't available."""
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
            if hasattr(self, tool_name):
                tool_func = getattr(self, tool_name)
                tool_result = tool_func(**params)
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
        Analyze a paper using LangChain.

        Args:
            paper_id: ID of the paper to analyze

        Returns:
            Analysis results
        """
        paper = self.db.get_paper(paper_id)
        if not paper:
            return {"error": "Paper not found"}

        return self.agent.analyze_paper(paper["title"], paper["abstract"])
