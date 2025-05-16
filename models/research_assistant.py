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
                logging.error("Failed to save PDF file")
                return {"success": False, "message": "Failed to save PDF file"}

            logging.info(f"PDF saved to {file_path}")

            # Extract metadata
            metadata = self.pdf_processor.extract_metadata(file_path)
            logging.info(f"Extracted metadata: {metadata}")

            # Extract full text
            full_text = self.pdf_processor.extract_full_text(file_path)
            logging.info(
                f"Extracted full text (length: {len(full_text) if full_text else 0})")

            # Store in database with full text
            paper_id = self.db.add_paper(
                title=metadata["title"],
                abstract=metadata["abstract"],
                authors=metadata["authors"],
                source="internal_upload",
                file_path=file_path,
                full_text=full_text
            )

            logging.info(f"Added paper to database with ID: {paper_id}")

            # Extract documents using LangChain
            documents = self.pdf_processor.extract_documents(file_path)
            logging.info(f"Extracted {len(documents)} document chunks")

            # Set document metadata
            for doc in documents:
                doc.metadata.update({
                    "doc_id": paper_id,
                    "title": metadata["title"],
                    "abstract": metadata["abstract"]
                })

            # Add to vector store using LangChain documents
            if documents:
                ids = self.vector_store.add_documents(documents)
                logging.info(f"Added {len(ids)} documents to vector store")
            else:
                logging.warning(
                    "No document chunks extracted, skipping vector store")

            # Update session memory
            self.session_memory["uploaded_papers"].append({
                "id": paper_id,
                "title": metadata["title"],
                "abstract": metadata["abstract"],
                "path": file_path,
                "has_full_text": bool(full_text)
            })

            return {
                "success": True,
                "paper_id": paper_id,
                "title": metadata["title"],
                "abstract": metadata["abstract"],
                "authors": metadata["authors"],
                "full_text_length": len(full_text) if full_text else 0
            }

        except Exception as e:
            logging.error(f"Error uploading paper: {str(e)}", exc_info=True)
            return {"success": False, "message": f"Upload failed: {str(e)}"}

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

        # Use full text if available, otherwise fall back to abstract
        if paper.get("full_text"):
            return self.agent.analyze_paper(paper["title"], paper["abstract"], paper["full_text"])
        else:
            return self.agent.analyze_paper(paper["title"], paper["abstract"])

    def chat_with_paper(self, paper_id: int, query: str) -> Dict:
        """
        Have a RAG-based conversation with a specific paper.

        Args:
            paper_id: ID of the paper to chat with
            query: User's question about the paper

        Returns:
            Dict with response and relevant source contexts
        """
        if not self.llm:
            return {"error": "LLM not initialized", "response": "Language model not available. Please check your API key."}

        try:
            # Get paper info
            paper = self.db.get_paper(paper_id)
            if not paper:
                return {"error": "Paper not found", "response": "Could not find the paper in the database."}

            # Create a retriever for this specific paper
            retriever = self.vector_store.db.as_retriever(
                search_kwargs={"k": 3, "filter": {"doc_id": paper_id}}
            )

            # Get relevant chunks
            docs = retriever.get_relevant_documents(query)

            # If no chunks are found, try using stored full text if available
            if not docs and paper.get('full_text'):
                # Create a prompt that uses the stored full text
                template = """You are an AI research assistant helping with questions about academic papers.
                Answer the question based on the paper information provided below. If the information
                needed to answer the question is not contained in the text, say "I don't have enough specific 
                information about that in this paper."
                
                Paper Title: {title}
                Paper Abstract: {abstract}
                
                Paper Full Text (excerpt):
                {full_text}
                
                Question: {question}
                """

                # Truncate full text if it's too long
                max_text_length = 30000
                full_text = paper.get('full_text', '')
                if len(full_text) > max_text_length:
                    full_text = full_text[:max_text_length] + \
                        "... [text truncated]"

                prompt = ChatPromptTemplate.from_template(template)

                chain = (
                    {"title": lambda _: paper.get('title', "Unknown Paper"),
                     "abstract": lambda _: paper.get('abstract', ""),
                     "full_text": lambda _: full_text,
                     "question": lambda x: x}
                    | prompt
                    | self.llm
                    | StrOutputParser()
                )

                response = chain.invoke(query)

                return {
                    "response": response,
                    "sources": [{"text": "Based on the full text of the paper", "metadata": {"note": "Using stored full text"}}]
                }

            # If no chunks found and no full text, fall back to using just metadata
            elif not docs:
                # Get the paper's full text if available but not stored in DB
                full_text = ""
                if paper.get('file_path') and os.path.exists(paper.get('file_path')):
                    try:
                        full_text = self.pdf_processor.extract_full_text(
                            paper.get('file_path'))
                    except Exception as e:
                        logging.error(f"Error extracting full text: {e}")

                # If we have some text to work with
                if full_text:
                    # Create a prompt that uses the paper title and abstract at minimum
                    template = """You are an AI research assistant helping with questions about academic papers.
                    
                    Paper Title: {title}
                    Paper Abstract: {abstract}
                    
                    You've been asked about this paper, but could only access limited information.
                    Do your best to answer based on the title and abstract, and explain what might be needed for a more complete answer.
                    
                    Question: {question}
                    """

                    prompt = ChatPromptTemplate.from_template(template)

                    chain = (
                        {"title": lambda _: paper.get('title', "Unknown Paper"),
                         "abstract": lambda _: paper.get('abstract', ""),
                         "question": lambda x: x}
                        | prompt
                        | self.llm
                        | StrOutputParser()
                    )

                    response = chain.invoke(query)

                    return {
                        "response": response,
                        "sources": [{"text": paper.get('abstract', ""), "metadata": {"note": "Only abstract available"}}]
                    }
                else:
                    # Minimal information case
                    return {
                        "response": f"I don't have enough information from '{paper.get('title')}' to answer your question. The paper may not have been fully processed or indexed correctly. Try uploading the paper again or reformulating your question.",
                        "sources": []
                    }

            # If chunks found, create prompt for RAG with the retrieved documents
            template = """You are an AI research assistant helping with questions about academic papers.
            Answer the question based ONLY on the context provided below. If you don't know or the answer
            is not in the context, say "I don't have enough information about that in this paper."
            
            Paper Title: {title}
            
            Context from the paper:
            {context}
            
            Question: {question}
            """

            prompt = ChatPromptTemplate.from_template(template)

            # Combine context from documents
            contexts = [doc.page_content for doc in docs]
            combined_context = "\n\n---\n\n".join(contexts)

            # Create and execute RAG chain
            chain = (
                {"context": lambda _: combined_context,
                 "question": lambda x: x,
                 "title": lambda _: paper.get('title', "Unknown Paper")}
                | prompt
                | self.llm
                | StrOutputParser()
            )

            response = chain.invoke(query)

            return {
                "response": response,
                "sources": [{"text": doc.page_content, "metadata": doc.metadata} for doc in docs]
            }

        except Exception as e:
            logging.error(f"Error in chat_with_paper: {e}")
            return {"error": str(e), "response": f"Error processing your question: {str(e)}"}

    def delete_paper(self, paper_id: int) -> Dict:
        """
        Delete a paper from the database and clean up associated resources.

        Args:
            paper_id: ID of the paper to delete

        Returns:
            Dict with success status and message
        """
        try:
            # Delete from database and get file path
            success, result = self.db.delete_paper(paper_id)

            if not success:
                return {"success": False, "message": result}

            file_path = result

            # Delete the associated PDF file if it exists
            if file_path and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception as e:
                    logging.warning(f"Could not delete file {file_path}: {e}")

            # Remove from vector store if present
            try:
                # Filter by doc_id to find and delete vector store entries for this paper
                if self.vector_store and self.vector_store.db:
                    # Get all documents and filter for the ones with this paper's ID
                    docs = self.vector_store.db.get()
                    if docs and "metadatas" in docs:
                        ids_to_delete = []
                        for i, metadata in enumerate(docs["metadatas"]):
                            if metadata.get("doc_id") == paper_id:
                                ids_to_delete.append(docs["ids"][i])

                        # Delete the documents from vector store if any found
                        if ids_to_delete:
                            self.vector_store.db.delete(ids_to_delete)
                            self.vector_store.db.persist()
            except Exception as e:
                logging.warning(f"Error cleaning up vector store entries: {e}")

            # Update session memory to remove the paper if present
            for i, paper in enumerate(self.session_memory["uploaded_papers"]):
                if paper.get("id") == paper_id:
                    self.session_memory["uploaded_papers"].pop(i)
                    break

            return {"success": True, "message": "Paper successfully deleted"}
        except Exception as e:
            logging.error(f"Error deleting paper: {e}")
            return {"success": False, "message": str(e)}
