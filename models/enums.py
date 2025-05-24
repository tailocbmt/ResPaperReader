from enum import Enum

ID = "id"
CREATED_AT = "created_at"
TITLE = "title"
ABSTRACT = "abstract"
AUTHORS = "authors"
SOURCE = "source"
URL_PATH = "url_path"
EMBEDDING = "embedding"
FULL_TEXT = "full_text"

COLUMNS = [
    ID,
    CREATED_AT,
    TITLE,
    ABSTRACT,
    AUTHORS,
    SOURCE,
    URL_PATH,
    EMBEDDING,
    FULL_TEXT
]

INTERNAL_SEARCH = "internal_search"
WEB_SEARCH = "web_search"
CONFERENCE_SEARCH = "conference_search"
COMPARE_PAPERS = "compare_papers"
AGENT_TOOLS = {
    INTERNAL_SEARCH: {
        "name": "internal_search",
        "description": "Search for papers in the internal database using keywords or semantic search",
        "parameters": {"query": "The search query"}
    },
    WEB_SEARCH: {
        "name": "web_search",
        "description": "Search for papers using external APIs like arXiv and Semantic Scholar",
        "parameters": {
            "query": "The search query",
            "source": "Optional source to search (arxiv or semantic_scholar)"
        }
    },
    CONFERENCE_SEARCH: {
        "name": "conference_search",
        "description": "Search for papers from a specific conference",
        "parameters": {
            "conference": "Conference name (e.g., ICLR, NeurIPS, ACL)",
            "year": "Optional year of the conference"
        }
    },
    COMPARE_PAPERS: {
        "name": "compare_papers",
        "description": "Compare two research papers and generate a structured report",
        "parameters": {
            "paper_id_1": "ID of the first paper to compare",
            "paper_id_2": "ID of the second paper to compare"
        }
    }
}


class NavigationType(Enum):
    CHAT_ASSISTANT = "Chat Assistant"
    UPLOAD_PAPERS = "Upload Papers"
    SEARCH_PAPERS = "Search Papers"
    PAPER_DATABASE = "Paper Database"
    CHAT_WITH_PAPERS = "Chat with Papers"


class PaperSource(Enum):
    USER = "user"
    INTERNET = "internet"


class LLMSource(Enum):
    GEMINI = "gemini"
    MISTRAL = "mistral"
    LLAMA2 = "llama2"


class WebSearchSource(Enum):
    ARXIV = "arxiv"
    SEMANTIC_SCHOLAR = "semantic_scholar"


class PromptFunctionType(Enum):
    ANALYZE_PAPER = "analyze_paper"
    COMPARE_PAPERS = "compare_papers"
    TOOL_ROUTER = "tool_router"


class PromptType(Enum):
    SYSTEM = "system"
    HUMAN = "human"
