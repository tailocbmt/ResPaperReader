import time
import arxiv
import logging
from typing import Dict, List
import semanticscholar as sch

from models.enums import WebSearchSource


class WebSearchManager:
    def __init__(self):
        """Initialize the API service for external paper sources."""
        self.semantic_client = sch.SemanticScholar()
        self.rate_limit_delay = 1  # seconds between requests

    def search_arxiv(
        self,
        query: str,
        limit: int = 5
    ) -> List[Dict]:
        """
        Search for papers on arXiv.

        Args:
            query: Search query
            max_results: Maximum number of results to return

        Returns:
            List of papers with metadata
        """
        try:
            # Configure search client
            search = arxiv.Search(
                query=query,
                max_results=limit,
                sort_by=arxiv.SortCriterion.Relevance
            )

            papers = []
            for result in search.results():
                papers.append(
                    {
                        'title': result.title,
                        'abstract': result.summary,
                        'authors': [author.name for author in result.authors],
                        'url': result.entry_id,
                        'pdf_url': result.pdf_url,
                        'source': WebSearchSource.ARXIV.value,
                        'published_at': result.published.strftime('%Y-%m-%d') if result.published else None
                    }
                )

            return papers
        except Exception as e:
            logging.error(f"Error searching arXiv: {e}")
            return []

    def search_semantic_scholar(
        self,
        query: str,
        limit: int = 5
    ) -> List[Dict]:
        """
        Search for papers on Semantic Scholar.

        Args:
            query: Search query
            max_results: Maximum number of results to return

        Returns:
            List of papers with metadata
        """
        try:
            papers = []

            # Use the Semantic Scholar API
            results = self.client.search_paper(query, limit=limit)
            time.sleep(self.rate_limit_delay)  # Respect rate limits

            for paper in results:
                # Get more details for this paper
                if paper.get('paperId'):
                    try:
                        details = self.semantic_client.get_paper(
                            paper.get('paperId')
                        )

                        time.sleep(self.rate_limit_delay)
                    except Exception as e:
                        logging.error(f"Error getting paper details: {e}")
                        details = paper
                else:
                    details = paper

                papers.append(
                    {
                        'title': details.get('title', 'Unknown Title'),
                        'abstract': details.get('abstract', ''),
                        'authors': [author.get('name', 'Unknown') for author in details.get('authors', [])],
                        'url': f"https://www.semanticscholar.org/paper/{details.get('paperId', '')}" if details.get('paperId') else None,
                        'source': WebSearchSource.SEMANTIC_SCHOLAR.value,
                        'year': details.get('year')
                    }
                )

            return papers
        except Exception as e:
            logging.error(f"Error searching Semantic Scholar: {e}")
            return []

    def search_papers(
        self,
        query: str,
        source: str = "Both",
        limit: str = 10
    ) -> List[Dict]:
        """
        Search for papers from a specific conference.

        Args:
            conf_name: Conference name (e.g., 'ICLR', 'NeurIPS', 'ACL')
            year: Publication year
            max_results: Maximum number of results to return

        Returns:
            List of papers with metadata
        """
        results = []
        if source == WebSearchSource.ARXIV.value or source == "Both":
            arxiv_results = self.search_arxiv(
                query=query
            )
            results.extend(arxiv_results)

        if source == WebSearchSource.SEMANTIC_SCHOLAR.value or source == "Both":
            semantic_scholar_results = self.search_semantic_scholar(
                query=query
            )
            results.extend(semantic_scholar_results)

        # Combine results (with deduplication based on title)
        seen_titles = set()
        combined_results = []

        # Process arXiv results first
        for paper in results:
            # Use first 100 chars of title as key
            title_key = paper['title'].lower()[:100]
            if title_key not in seen_titles:
                seen_titles.add(title_key)
                combined_results.append(paper)

        return combined_results[:limit]
