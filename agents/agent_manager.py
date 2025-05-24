import os
import json
import logging
from typing import List, Dict, Any, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama

from utility.helpers import load_prompts
from models.enums import AGENT_TOOLS, LLMSource, PromptFunctionType, PromptType


class AgentManager:
    def __init__(
        self,
        google_api_key: str = None,
        llm_name: str = LLMSource.GEMINI.value,
        temperature: float = 0.3
    ):
        self.google_api_key = google_api_key or os.environ.get(
            "GEMINI_API_KEY")
        self.llm_name = llm_name
        self.temperature = temperature
        self.initialize_llm(
            llm_name=self.llm_name,
            temperature=self.temperature
        )

        self.conversation_history = []
        self.system_prompts = load_prompts()

    def load_llm(self, llm_name: str, temperature: float = 0.3):
        if llm_name == LLMSource.GEMINI.name:
            return ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                temperature=temperature,
                google_api_key=self.api_key
            )
        elif llm_name in {LLMSource.MISTRAL.name, LLMSource.LLAMA2.name}:
            return ChatOllama(
                model=llm_name,
                temperature=temperature
            )
        else:
            raise ValueError(f"Unsupported model: {llm_name}")

    def initialize_llm(self, llm_name: str, temperature: float = 0.3):
        try:
            self.model = self.load_llm(llm_name, temperature)
            self.parser = StrOutputParser()
        except Exception as e:
            logging.error(f"Error configuring LLM '{llm_name}': {e}")
            self.model = None

    def run_paper_analysis(
            self,
            title: str,
            abstract: str,
            full_text: str = None
    ):
        if not self.model:
            return {"error": f"{self.llm_name} model not configured."}

        paper_content = self._build_paper_content(title, abstract, full_text)

        template = self.system_prompts.get(
            PromptFunctionType.ANALYZE_PAPER.value, {})
        system_msg = template.get(
            PromptType.SYSTEM.value, "You are a research assistant.")
        human_msg = template.get(
            PromptType.HUMAN.value, "").replace(
            "{content}", paper_content)

        system_msg = self.system_prompts.get(
            "analyze_paper", "You are a research assistant.")

        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    content=system_msg),
                HumanMessage(
                    content=human_msg)
            ]
        )

        try:
            analysis_chain = prompt | self.model | self.parser
            result = analysis_chain.invoke({})
            return {"analysis": result, "title": title}
        except Exception as e:
            logging.error(f"Error generating paper analysis: {e}")
            return {"error": str(e)}

    def run_paper_comparison(self, paper1: Dict, paper2: Dict):
        if not self.model:
            return f"{self.llm_name} model not configured."

        content1 = self._format_paper_content(paper1, max_length=15000)
        content2 = self._format_paper_content(paper2, max_length=15000)

        template = self.system_prompts.get(
            PromptFunctionType.COMPARE_PAPERS.value, {})
        system_msg = template.get(
            PromptType.SYSTEM.value, "You are a comparison agent.")
        human_msg = template.get(
            PromptType.HUMAN.value, "").replace(
            "{paper1}", content1).replace("{paper2}", content2)

        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    content=system_msg),
                HumanMessage(
                    content=human_msg)

            ]
        )

        try:
            comparison_chain = prompt | self.model | self.parser
            return comparison_chain.invoke({})
        except Exception as e:
            logging.error(f"Error generating paper comparison: {e}")
            return f"Failed to generate comparison: {str(e)}"

    def process_query(self, query: str, additional_context: str = None):
        if not self.model:
            return f"{self.llm_name} not configured. Please configure."

        tool_doc = json.dumps(AGENT_TOOLS, indent=2)
        formatted_context = self._format_context_dict(additional_context)

        template = self.system_prompts.get(
            PromptFunctionType.TOOL_ROUTER.value, {})
        system_msg = template.get(
            PromptType.SYSTEM.value, "").replace(
            "{tool_docs}", tool_doc)
        human_msg = template.get(
            PromptType.HUMAN.value, "").replace(
            "{query}", query).replace("{context}", formatted_context)

        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    content=system_msg),
                HumanMessage(
                    content=human_msg)
            ]
        )

        try:
            query_chain = prompt | self.model | self.parser
            agent_reply = query_chain.invoke({})

            self.conversation_history.append(
                {"user": query, "assistant": agent_reply})
            return agent_reply
        except Exception as e:
            logging.error(f"Error processing query: {e}")
            return f"Encountered an error: {str(e)}"

    def _build_paper_content(self, title: str, abstract: str, full_text: str = None) -> str:
        content = f"Title: {title}\n\nAbstract: {abstract}"
        if full_text:
            truncated = full_text[:30000]
            if len(full_text) > 30000:
                truncated += "... [text truncated]"
            content += f"\n\nFull Text Excerpt:\n{truncated}"
        return content

    def _format_paper_content(self, paper: Dict, max_length: int = 15000) -> str:
        content = f"Title: {paper.get('title', 'Not available')}\nAbstract: {paper.get('abstract', 'Not available')}"
        full_text = paper.get("full_text")
        if full_text:
            truncated = full_text[:max_length]
            if len(full_text) > max_length:
                truncated += "... [text truncated]"
            content += f"\n\nFull Text Excerpt:\n{truncated}"
        return content

    def _format_context_dict(self, context_data: Any) -> str:
        if not context_data:
            return ""
        try:
            return f"\nAdditional context:\n{json.dumps(context_data, indent=2)}"
        except Exception:
            return f"\nAdditional context: {str(context_data)}"
