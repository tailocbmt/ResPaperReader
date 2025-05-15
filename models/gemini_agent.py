import google.generativeai as genai
import os
import json
import logging
from typing import List, Dict, Any


class GeminiAgent:
    def __init__(self, api_key=None):
        """Initialize the Gemini agent with API key."""
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.configure()
        self.conversation_history = []
        self.available_tools = self._define_tools()

    def configure(self):
        """Configure the Gemini API."""
        try:
            genai.configure(api_key=self.api_key)
            # Updated to use gemini-2.0-flash instead of gemini-pro
            self.model = genai.GenerativeModel('gemini-2.0-flash')
            logging.info(
                "Successfully configured Gemini API with model: gemini-2.0-flash")
        except Exception as e:
            logging.error(f"Error configuring Gemini API: {e}")
            self.model = None

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
        Analyze a research paper and extract key information.

        Args:
            title: Paper title
            abstract: Paper abstract
            full_text: Optional full text of the paper

        Returns:
            Dict with analysis results
        """
        if not self.model or not self.api_key:
            return {"error": "Gemini API not configured"}

        prompt = f"""Analyze the following research paper and extract key information:
Title: {title}

Abstract: {abstract}

Please provide:
1. Research problem/question
2. Key methods
3. Main contributions
4. 3-5 keywords

Format your response as a structured report.
"""

        try:
            response = self.model.generate_content(prompt)
            return {
                "analysis": response.text,
                "title": title
            }
        except Exception as e:
            logging.error(f"Error generating paper analysis: {e}")
            return {"error": str(e)}

    def compare_papers(self, paper1, paper2):
        """
        Compare two research papers and generate a structured report.

        Args:
            paper1: Dict containing first paper metadata
            paper2: Dict containing second paper metadata

        Returns:
            Comparison report
        """
        if not self.model or not self.api_key:
            return "Gemini API not configured. Please add your API key."

        prompt = f"""Compare the following two research papers:

Paper 1:
Title: {paper1.get('title', 'Unknown')}
Abstract: {paper1.get('abstract', 'Not available')}

Paper 2:
Title: {paper2.get('title', 'Unknown')}
Abstract: {paper2.get('abstract', 'Not available')}

Please provide a structured comparison including:
1. Research goals
2. Methodologies
3. Key contributions
4. Strengths and weaknesses
5. Similarities between the papers
6. Major differences

Format your response as a structured report with clear headings and bullet points.
"""

        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            logging.error(f"Error generating paper comparison: {e}")
            return f"Failed to generate comparison: {str(e)}"

    def process_query(self, query, context=None):
        """
        Process a user query using the ReAct approach.

        Args:
            query: User's natural language query
            context: Additional context for the query

        Returns:
            Response from the agent
        """
        if not self.model or not self.api_key:
            return "Gemini API not configured. Please add your API key."

        context_str = ""
        if context:
            context_str = f"\nAdditional context:\n{json.dumps(context, indent=2)}"

        prompt = f"""{self.get_system_prompt()}

User query: {query}{context_str}

Based on this query, what action should be taken and what information should be provided?
"""

        try:
            response = self.model.generate_content(prompt)

            # Add to conversation history
            self.conversation_history.append({
                "user": query,
                "assistant": response.text
            })

            return response.text
        except Exception as e:
            logging.error(f"Error processing query: {e}")
            return f"I encountered an error: {str(e)}"
