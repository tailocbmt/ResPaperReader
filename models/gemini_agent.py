import os
import json
import logging
from typing import List, Dict, Any

# LangChain imports
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.agents import Tool, create_structured_chat_agent
from langchain.agents.agent import AgentExecutor


class GeminiAgent:
    def __init__(self, api_key=None):
        """Initialize the Gemini agent with API key using LangChain."""
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.configure()
        self.conversation_history = []
        self.available_tools = self._define_tools()

    def configure(self):
        """Configure the LangChain Gemini integration."""
        try:
            os.environ["GOOGLE_API_KEY"] = self.api_key
            # Using LangChain's wrapper for Gemini
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash", temperature=0.2)
            self.output_parser = StrOutputParser()
            logging.info(
                "Successfully configured Gemini API via LangChain with model: gemini-2.0-flash")
        except Exception as e:
            logging.error(f"Error configuring Gemini API via LangChain: {e}")
            self.llm = None

    def _define_tools(self):
        """Define the available tools for the agent."""
        return {
            "internal_search": {
                "name": "internal_search",
                "description": "Search for papers in the internal database using keywords or semantic search",
                "parameters": {
                    "query": "The search query"
                }
            },
            "web_search": {
                "name": "web_search",
                "description": "Search for papers using external APIs like arXiv and Semantic Scholar",
                "parameters": {
                    "query": "The search query",
                    "source": "Optional source to search (arxiv or semantic_scholar)"
                }
            },
            "conference_search": {
                "name": "conference_search",
                "description": "Search for papers from a specific conference",
                "parameters": {
                    "conference": "Conference name (e.g., ICLR, NeurIPS, ACL)",
                    "year": "Optional year of the conference"
                }
            },
            "compare_papers": {
                "name": "compare_papers",
                "description": "Compare two research papers and generate a structured report",
                "parameters": {
                    "paper_id_1": "ID of the first paper to compare",
                    "paper_id_2": "ID of the second paper to compare"
                }
            }
        }

    def get_system_prompt(self):
        """Get the system prompt for the agent."""
        return f"""You are a Research Paper Assistant that helps users find, analyze, and compare research papers.
You can understand natural language commands and determine which tools to use.

Available tools:
{json.dumps(self.available_tools, indent=2)}

For each user query, analyze what they're asking for and choose the appropriate tool.
Format your responses in a clear, informative way suitable for academic research.
If you need to use a tool, output it in the format: <tool>tool_name(parameters)</tool>
"""

    def analyze_paper(self, title, abstract, full_text=None):
        """
        Analyze a research paper and extract key information using LangChain.

        Args:
            title: Paper title
            abstract: Paper abstract
            full_text: Optional full text of the paper

        Returns:
            Dict with analysis results
        """
        if not self.llm or not self.api_key:
            return {"error": "Gemini API not configured"}

        # Include full text in analysis if available
        content = f"Title: {title}\n\nAbstract: {abstract}"

        if full_text:
            # If we have full text, provide a comprehensive section of it to the model
            # Limit the length to avoid token limits
            max_text_length = 30000  # Adjust based on model's context window
            truncated_text = full_text[:max_text_length]
            if len(full_text) > max_text_length:
                truncated_text += "... [text truncated]"

            content += f"\n\nFull Text Excerpt:\n{truncated_text}"

        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(
                content="You are a research paper analysis assistant that extracts key information from papers."),
            HumanMessage(content=f"""Analyze the following research paper and extract key information:
{content}

Please provide:
1. Research problem/question
2. Key methods
3. Main contributions
4. Findings and conclusions
5. 3-5 keywords
6. Implications for the field

Format your response as a structured report with clear headings.
""")
        ])

        try:
            chain = prompt | self.llm | self.output_parser
            analysis = chain.invoke({})
            return {
                "analysis": analysis,
                "title": title
            }
        except Exception as e:
            logging.error(f"Error generating paper analysis: {e}")
            return {"error": str(e)}

    def compare_papers(self, paper1, paper2):
        """
        Compare two research papers and generate a structured report using LangChain.

        Args:
            paper1: Dict containing first paper metadata
            paper2: Dict containing second paper metadata

        Returns:
            Comparison report
        """
        if not self.llm or not self.api_key:
            return "Gemini API not configured. Please add your API key."

        # Prepare content for paper 1
        paper1_content = f"Title: {paper1.get('title', 'Unknown')}\nAbstract: {paper1.get('abstract', 'Not available')}"
        if paper1.get('full_text'):
            # Limit the length to prevent token overflow
            max_text_length = 15000  # Reduced size to accommodate both papers
            truncated_text = paper1.get('full_text')[:max_text_length]
            if len(paper1.get('full_text', '')) > max_text_length:
                truncated_text += "... [text truncated]"
            paper1_content += f"\n\nFull Text Excerpt:\n{truncated_text}"

        # Prepare content for paper 2
        paper2_content = f"Title: {paper2.get('title', 'Unknown')}\nAbstract: {paper2.get('abstract', 'Not available')}"
        if paper2.get('full_text'):
            # Limit the length to prevent token overflow
            max_text_length = 15000  # Reduced size to accommodate both papers
            truncated_text = paper2.get('full_text')[:max_text_length]
            if len(paper2.get('full_text', '')) > max_text_length:
                truncated_text += "... [text truncated]"
            paper2_content += f"\n\nFull Text Excerpt:\n{truncated_text}"

        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(
                content="You are a research paper comparison assistant with expertise in analyzing academic papers."),
            HumanMessage(content=f"""Compare the following two research papers based on all provided information:

Paper 1:
{paper1_content}

Paper 2:
{paper2_content}

Please provide a structured comparison including:
1. Research goals and objectives
2. Methodologies and approaches
3. Key contributions and innovations
4. Main findings and results
5. Strengths and limitations of each paper
6. Significant similarities between the papers
7. Important differences and contrasting aspects
8. Recommendations for which paper might be more relevant for different research contexts

Format your response as a structured report with clear headings and bullet points.
""")
        ])

        try:
            chain = prompt | self.llm | self.output_parser
            return chain.invoke({})
        except Exception as e:
            logging.error(f"Error generating paper comparison: {e}")
            return f"Failed to generate comparison: {str(e)}"

    def process_query(self, query, context=None):
        """
        Process a user query using LangChain.

        Args:
            query: User's natural language query
            context: Additional context for the query

        Returns:
            Response from the agent
        """
        if not self.llm or not self.api_key:
            return "Gemini API not configured. Please add your API key."

        context_str = ""
        if context:
            context_str = f"\nAdditional context:\n{json.dumps(context, indent=2)}"

        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=self.get_system_prompt()),
            HumanMessage(
                content=f"User query: {query}{context_str}\n\nBased on this query, what action should be taken and what information should be provided?")
        ])

        try:
            chain = prompt | self.llm | self.output_parser
            response = chain.invoke({})

            # Add to conversation history
            self.conversation_history.append({
                "user": query,
                "assistant": response
            })

            return response
        except Exception as e:
            logging.error(f"Error processing query: {e}")
            return f"I encountered an error: {str(e)}"
