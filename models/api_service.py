import arxiv
import semanticscholar as sch
import logging
from datetime import datetime, timedelta
import time


class APIService:
    def __init__(self):
        """Initialize the API service for external paper sources."""
        self.client = sch.SemanticScholar()
        self.rate_limit_delay = 1  # seconds between requests

    def search_arxiv(self, query, max_results=5):
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
                max_results=max_results,
                sort_by=arxiv.SortCriterion.SubmittedDate
            )

            papers = []
            for result in search.results():
                papers.append({
                    'title': result.title,
                    'abstract': result.summary,
                    'authors': [author.name for author in result.authors],
                    'url': result.entry_id,
                    'pdf_url': result.pdf_url,
                    'source': 'arxiv',
                    'published': result.published.strftime('%Y-%m-%d') if result.published else None
                })

            return papers
        except Exception as e:
            logging.error(f"Error searching arXiv: {e}")
            return []

    def search_semantic_scholar(self, query, max_results=5):
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
            results = self.client.search_paper(query, limit=max_results)
            time.sleep(self.rate_limit_delay)  # Respect rate limits

            for paper in results:
                # Get more details for this paper
                if paper.get('paperId'):
                    try:
                        details = self.client.get_paper(paper.get('paperId'))
                        # Respect rate limits
                        time.sleep(self.rate_limit_delay)
                    except Exception as e:
                        logging.error(f"Error getting paper details: {e}")
                        details = paper
                else:
                    details = paper

                papers.append({
                    'title': details.get('title', 'Unknown Title'),
                    'abstract': details.get('abstract', ''),
                    'authors': [author.get('name', 'Unknown') for author in details.get('authors', [])],
                    'url': f"https://www.semanticscholar.org/paper/{details.get('paperId', '')}" if details.get('paperId') else None,
                    'source': 'semantic_scholar',
                    'year': details.get('year')
                })

            return papers
        except Exception as e:
            logging.error(f"Error searching Semantic Scholar: {e}")
            return []

    def search_papers_by_conference(self, conf_name, year=None, max_results=10):
        """
        Search for papers from a specific conference.

        Args:
            conf_name: Conference name (e.g., 'ICLR', 'NeurIPS', 'ACL')
            year: Publication year
            max_results: Maximum number of results to return

        Returns:
            List of papers with metadata
        """
        # If year is not specified, use current year
        if not year:
            year = datetime.now().year

        # Format query for arXiv
        query = f"{conf_name} {year}"

        # Get results from both sources
        arxiv_results = self.search_arxiv(query, max_results=max_results)
        scholar_results = self.search_semantic_scholar(
            query, max_results=max_results)

        # Combine results (with deduplication based on title)
        seen_titles = set()
        combined_results = []

        # Process arXiv results first
        for paper in arxiv_results:
            # Use first 100 chars of title as key
            title_key = paper['title'].lower()[:100]
            if title_key not in seen_titles:
                seen_titles.add(title_key)
                combined_results.append(paper)

        # Add unique Semantic Scholar results
        for paper in scholar_results:
            title_key = paper['title'].lower()[:100]
            if title_key not in seen_titles:
                seen_titles.add(title_key)
                combined_results.append(paper)

        return combined_results[:max_results]
