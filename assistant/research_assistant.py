from datetime import datetime
import os
import logging
import re
import json
from typing import List, Dict

from models.enums import COMPARE_PAPERS, CONFERENCE_SEARCH, INTERNAL_SEARCH, WEB_SEARCH, LLMSource, PaperSource, PromptType, WebSearchSource
from models.sql_client import SQLClient
from utility.helpers import load_prompts
from models.vector_store import VectorStore
from models.pdf_processor import PDFProcessor
from models.web_search_manager import WebSearchManager
from agents.agent_manager import AgentManager

# LangChain imports
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser


class ResearchAssistant:
    def __init__(
            self,
            gemini_api_key: str = None,
            llm_name: str = LLMSource.GEMINI.value):
        self.db = SQLClient(
            db_config={
                "db_name": os.environ.get("DB_NAME"),
                "db_host": os.environ.get("DB_HOST"),
                "db_user": os.environ.get("DB_USER"),
                "db_pass": os.environ.get("DB_PASS"),
                "db_port": os.environ.get("DB_PORT")
            }
        )
        self.vector = VectorStore()
        self.pdf = PDFProcessor()
        self.web_search = WebSearchManager()
        self.api_key = gemini_api_key or os.environ.get("GEMINI_API_KEY")
        self.prompts = load_prompts(
            file_path="prompts/assistant_prompts.json"
        )
        self.llm_name = llm_name
        self.llm_agent = AgentManager(
            google_api_key=self.api_key,
            llm_name=self.llm_name
        )

        try:
            self.model = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash", temperature=0.2, google_api_key=self.api_key
            )
        except Exception as e:
            logging.error(f"LLM init error: {e}")
            self.model = None

        self.session = {
            "uploaded_papers": [],
            "search_results": [],
            "last_comparison": None
        }

        self._init_tools()
        self._init_agent()

    def _init_tools(self):
        self.tools = [
            Tool(name=INTERNAL_SEARCH, func=self.search_internal,
                 description="Search internal DB"),
            Tool(name=WEB_SEARCH, func=self.search_web,
                 description="Search external sources"),
            Tool(name=CONFERENCE_SEARCH, func=self.search_conference,
                 description="Search by conference"),
            Tool(name=COMPARE_PAPERS, func=self.compare_papers,
                 description="Compare two papers")
        ]

    def _init_agent(self):
        if not self.model:
            self.agent_executor = None
            return

        try:
            template = self.prompts.get(
                "init_agent", {})
            system_msg = template.get(
                PromptType.SYSTEM.value, "You are a research paper assistant.")
            human_msg = template.get(
                PromptType.HUMAN.value, "")

            prompt = ChatPromptTemplate.from_messages(
                [
                    SystemMessage(content=system_msg),
                    MessagesPlaceholder(variable_name="tools"),
                    MessagesPlaceholder(variable_name="tool_names"),
                    MessagesPlaceholder(variable_name="chat_history"),
                    HumanMessage(content=human_msg),
                    MessagesPlaceholder(variable_name="agent_scratchpad")
                ]
            )

            agent = create_react_agent(self.model, self.tools, prompt)
            self.agent_executor = AgentExecutor(
                agent=agent, tools=self.tools, verbose=True)

        except Exception as e:
            logging.error(f"Agent init error: {e}")
            self.agent_executor = None

    def upload_paper(self, pdf_file):
        try:
            path = self.pdf.save_pdf(uploaded_file=pdf_file)
            if not path:
                return {"success": False, "message": "Save failed"}

            meta = self.pdf.extract_metadata(path)
            content = self.pdf.extract_full_text(path)

            paper_id = self.db.add_paper(
                title=meta["title"], abstract=meta["abstract"], authors=meta["authors"],
                source=PaperSource.USER.value, url_path=path, full_text=content
            )

            docs = self.pdf.extract_documents(path)
            for d in docs:
                d.metadata.update(
                    {"doc_id": paper_id, "title": meta["title"], "abstract": meta["abstract"]})

            if docs:
                self.vector.add_documents(docs)

            self.session["uploaded_papers"].append({
                "id": paper_id, "title": meta["title"], "abstract": meta["abstract"],
                "path": path, "has_full_text": bool(content)
            })

            return {
                "success": True, "paper_id": paper_id, "title": meta["title"],
                "abstract": meta["abstract"], "authors": meta["authors"],
                "full_text_length": len(content) if content else 0
            }

        except Exception as e:
            logging.error(f"Upload error: {str(e)}", exc_info=True)
            return {"success": False, "message": f"Upload failed: {str(e)}"}

    def search_internal(self, query: str) -> List[Dict]:
        matches = self.vector.search(query=query, top_k=3)
        results = []
        for m in matches:
            if m.get('doc_id'):
                paper = self.db.get_paper_by_id(paper_id=m['doc_id'])
                if paper:
                    paper['score'] = m.get('score', 0)
                    results.append(paper)

        self.session["search_results"] = results
        return results

    def search_web(self, query: str, source: str = "Both") -> List[Dict]:
        results = self.web_search.search_papers(query=query, source=source)
        self.session["search_results"] = results
        return results

    def search_conference(self, conference: str, year: str = None) -> List[Dict]:
        year = year or datetime.now().year
        query = f"{conference} {year}"
        results = self.web_search.search_papers(query=query)
        self.session["search_results"] = results
        return results

    def compare_papers(self, paper_1: Dict, paper_2: Dict) -> str:
        if not paper_1 or not paper_2:
            return "Missing one or both papers."

        report = self.llm_agent.run_paper_comparison(
            paper1=paper_1, paper2=paper_2)
        self.session["last_comparison"] = {
            "papers": [paper_1, paper_2], "report": report}
        return report

    def _legacy_process_query(self, user_query: str) -> Dict:
        context_info = {
            "uploaded_papers": [p.get("title") for p in self.session["uploaded_papers"]],
            "recent_searches": [p.get("title") for p in self.session["search_results"]]
        }

        response_text = self.llm_agent.process_query(
            query=user_query,
            additional_context=context_info
        )

        tool_tag_pattern = r"<tool>(\w+)\((.*?)\)</tool>"
        param_pattern = r'(\w+):\s*"([^"]*)"'
        tool_matches = re.findall(tool_tag_pattern, response_text)

        tool_outputs = []
        for tool_name, raw_param_str in tool_matches:
            param_dict = dict(re.findall(param_pattern, raw_param_str))
            if hasattr(self, tool_name):
                tool_func = getattr(self, tool_name)
                result = tool_func(**param_dict)
                tool_outputs.append({"tool": tool_name, "result": result})

        if tool_outputs:
            return {
                "action": "tool_results",
                "results": tool_outputs,
                "message": re.sub(tool_tag_pattern, "", response_text)
            }
        return {"action": "response", "message": response_text}

    def process_normal_query(self, user_query: str) -> Dict:
        if user_query.lower().startswith("upload"):
            return {
                "action": "upload_prompt",
                "message": "Please upload a PDF file to continue."
            }

        if not self.agent_executor:
            return self._legacy_process_query(user_query)

        try:
            context_bundle = {
                "uploaded_papers": [p.get("title") for p in self.session["uploaded_papers"]],
                "recent_searches": [p.get("title") for p in self.session["search_results"]]
            }
            prompt_input = f"{user_query}\n\nContext: {json.dumps(context_bundle)}"
            prompt_input
            logging.error(f"PROMPT INPUTTT: {prompt_input}")
            agent_output = self.agent_executor.invoke(
                {"input": prompt_input, "chat_history": []})
            message_text = agent_output["output"]

            if "intermediate_steps" in agent_output:
                tool_execs = [
                    {"tool": step[0].tool, "result": step[1]}
                    for step in agent_output["intermediate_steps"]
                    if isinstance(step[1], list) and step[1]
                ]
                if tool_execs:
                    return {
                        "action": "tool_results",
                        "results": tool_execs,
                        "message": message_text
                    }

            return {
                "action": "response",
                "message": message_text
            }

        except Exception as e:
            logging.error(f"LangChain agent error: {e}")
            return self._legacy_process_query(user_query)

    def analyze_paper(self, paper_id: int) -> Dict:
        paper_data = self.db.get_paper_by_id(paper_id)
        if not paper_data:
            return {"error": "Paper not exist"}

        return self.llm_agent.run_paper_analysis(
            title=paper_data["title"],
            abstract=paper_data["abstract"],
            full_text=paper_data.get("full_text")
        )

    def chat_with_paper(self, paper_id: int, user_question: str) -> Dict:
        if not self.model:
            return {"error": "LLM not initialized", "response": "Language model not available."}

        try:
            paper_record = self.db.get_paper_by_id(paper_id)
            if not paper_record:
                return {"error": "Paper not found", "response": "Could not find the paper in the database."}

            retriever = self.vector.db.as_retriever(
                search_kwargs={"k": 3, "filter": {"doc_id": paper_id}})
            matched_chunks = retriever.get_relevant_documents(user_question)

            template = self.prompts.get(
                "chat_with_paper", {})
            full_text_msg = template.get(
                "w_full_text")
            wo_full_text_msg = template.get(
                "wo_full_text")
            limited_text_msg = template.get(
                "limited_text")

            if not matched_chunks and paper_record.get("full_text"):
                full_text = paper_record["full_text"][:30000] + (
                    "... [text truncated]" if len(paper_record["full_text"]) > 30000 else "")

                prompt = ChatPromptTemplate.from_template(full_text_msg)

                chain = (
                    {"title": lambda _: paper_record.get("title", "Unknown Paper"),
                     "abstract": lambda _: paper_record.get("abstract", ""),
                     "full_text": lambda _: full_text,
                     "question": lambda x: x}
                    | prompt
                    | self.model
                    | StrOutputParser()
                )

                response = chain.invoke(user_question)
                return {"response": response, "sources": [{"text": "Based on the full text of the paper", "metadata": {"note": "Using stored full text"}}]}

            elif not matched_chunks:
                recovered_text = ""
                if paper_record.get("file_path") and os.path.exists(paper_record["file_path"]):
                    try:
                        recovered_text = self.pdf.extract_full_text(
                            paper_record["file_path"])
                    except Exception as e:
                        logging.error(f"Text extraction error: {e}")

                if recovered_text:
                    prompt = ChatPromptTemplate.from_template(limited_text_msg)

                    chain = (
                        {"title": lambda _: paper_record.get("title", "Unknown Paper"),
                         "abstract": lambda _: paper_record.get("abstract", ""),
                         "question": lambda x: x}
                        | prompt
                        | self.model
                        | StrOutputParser()
                    )

                    response = chain.invoke(user_question)
                    return {"response": response, "sources": [{"text": paper_record.get("abstract", ""), "metadata": {"note": "Only abstract available"}}]}

                return {"response": f"I don't have enough information from '{paper_record.get('title')}' to answer your question. Try uploading again or reformulate the question.", "sources": []}

            context_blocks = [doc.page_content for doc in matched_chunks]
            full_context = "\n\n---\n\n".join(context_blocks)

            prompt = ChatPromptTemplate.from_template(wo_full_text_msg)

            chain = (
                {"title": lambda _: paper_record.get("title", "Unknown Paper"),
                 "context": lambda _: full_context,
                 "question": lambda x: x}
                | prompt
                | self.model
                | StrOutputParser()
            )

            response = chain.invoke(user_question)

            return {
                "response": response,
                "sources": [{"text": doc.page_content, "metadata": doc.metadata} for doc in matched_chunks]
            }

        except Exception as err:
            logging.error(f"chat_with_paper error: {err}")
            return {"error": str(err), "response": f"Error processing your question: {str(err)}"}

    def delete_paper(self, paper_id: int) -> Dict:
        try:
            deletion_success, result_path = self.db.delete_paper(paper_id)
            if not deletion_success:
                return {"success": False, "message": result_path}

            if result_path and os.path.exists(result_path):
                try:
                    os.remove(result_path)
                except Exception as err:
                    logging.warning(
                        f"File removal failed for {result_path}: {err}")

            try:
                vector_data = self.vector.db.get()
                if vector_data and "metadatas" in vector_data:
                    ids_to_delete = [
                        vector_data["ids"][i]
                        for i, metadata in enumerate(vector_data["metadatas"])
                        if metadata.get("doc_id") == paper_id
                    ]
                    if ids_to_delete:
                        self.vector.db.delete(ids_to_delete)
                        self.vector.db.persist()
            except Exception as err:
                logging.warning(f"Vector store cleanup failed: {err}")

            self.session["uploaded_papers"] = [
                p for p in self.session["uploaded_papers"] if p.get("id") != paper_id
            ]

            return {"success": True, "message": "Successfully deleted"}

        except Exception as e:
            logging.error(f"Error deleting paper: {e}")
            return {"success": False, "message": str(e)}
