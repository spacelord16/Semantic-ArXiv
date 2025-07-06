import requests
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional
import time
from datetime import datetime
import re
from urllib.parse import quote
from config import config


class ArxivPaper:
    """Data class for arXiv paper information"""

    def __init__(
        self,
        id: str,
        title: str,
        authors: List[str],
        abstract: str,
        published: str,
        url: str,
        categories: List[str],
    ):
        self.id = id
        self.title = title
        self.authors = authors
        self.abstract = abstract
        self.published = published
        self.url = url
        self.categories = categories

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "title": self.title,
            "authors": self.authors,
            "abstract": self.abstract,
            "published": self.published,
            "url": self.url,
            "categories": self.categories,
        }


class ArxivClient:
    """Client for interacting with the arXiv API"""

    def __init__(self):
        self.base_url = config.ARXIV_API_BASE_URL
        self.max_results_per_request = 1000  # arXiv API limit

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content"""
        if not text:
            return ""

        # Remove extra whitespace and newlines
        text = re.sub(r"\s+", " ", text.strip())
        # Keep most characters, just remove problematic ones
        text = re.sub(r'[^\w\s\-.,;:!?()[\]{}"\'/&@+=%]', "", text)
        return text

    def _parse_authors(self, authors_elem) -> List[str]:
        """Parse author information from XML"""
        authors = []
        for author in authors_elem:
            name_elem = author.find("{http://www.w3.org/2005/Atom}name")
            if name_elem is not None and name_elem.text:
                authors.append(self._clean_text(name_elem.text))
        return authors

    def _parse_categories(self, entry) -> List[str]:
        """Parse category information from XML"""
        categories = []
        for category in entry.findall("{http://www.w3.org/2005/Atom}category"):
            term = category.get("term")
            if term:
                categories.append(term)
        return categories

    def _parse_entry(self, entry) -> Optional[ArxivPaper]:
        """Parse a single paper entry from XML"""
        try:
            # Get basic information
            id_elem = entry.find("{http://www.w3.org/2005/Atom}id")
            title_elem = entry.find("{http://www.w3.org/2005/Atom}title")
            summary_elem = entry.find("{http://www.w3.org/2005/Atom}summary")
            published_elem = entry.find("{http://www.w3.org/2005/Atom}published")
            authors_elem = entry.findall("{http://www.w3.org/2005/Atom}author")

            # Check if elements exist and have text content
            missing = []
            if id_elem is None or not id_elem.text:
                missing.append("id")
            if title_elem is None or not title_elem.text:
                missing.append("title")
            if summary_elem is None or not summary_elem.text:
                missing.append("summary")
            if published_elem is None or not published_elem.text:
                missing.append("published")

            if missing:
                return None

            # Extract and clean data
            paper_id = id_elem.text.split("/")[-1] if id_elem.text else ""
            title = self._clean_text(title_elem.text)
            abstract = self._clean_text(summary_elem.text)
            published = published_elem.text
            authors = self._parse_authors(authors_elem)
            categories = self._parse_categories(entry)
            url = id_elem.text

            # Validate required fields
            if not all([paper_id, title, abstract]):
                return None
            return ArxivPaper(
                id=paper_id,
                title=title,
                authors=authors,
                abstract=abstract,
                published=published,
                url=url,
                categories=categories,
            )

        except Exception as e:
            print(f"❌ Error parsing entry: {e}")
            import traceback

            traceback.print_exc()
            return None

    def search_papers(
        self, query: str, max_results: int = 100, start: int = 0
    ) -> List[ArxivPaper]:
        """
        Search for papers using the arXiv API

        Args:
            query: Search query (can include categories, keywords, etc.)
            max_results: Maximum number of results to return
            start: Starting index for pagination

        Returns:
            List of ArxivPaper objects
        """
        papers = []

        # Construct API URL
        params = {
            "search_query": query,
            "start": start,
            "max_results": min(max_results, self.max_results_per_request),
        }

        url = f"{self.base_url}/query"

        try:
            print(f"Fetching papers: {query} (start={start}, max={max_results})")
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()

            # Parse XML response
            root = ET.fromstring(response.content)
            entries = root.findall("{http://www.w3.org/2005/Atom}entry")

            for entry in entries:
                paper = self._parse_entry(entry)
                if paper:
                    papers.append(paper)

            print(f"Successfully fetched {len(papers)} papers")

            # Rate limiting - be nice to arXiv
            time.sleep(0.5)

        except requests.RequestException as e:
            print(f"Error fetching papers: {e}")
        except ET.ParseError as e:
            print(f"Error parsing XML: {e}")

        return papers

    def fetch_papers_by_category(
        self, category: str, max_papers: int = 1000
    ) -> List[ArxivPaper]:
        """
        Fetch papers from a specific arXiv category

        Args:
            category: arXiv category (e.g., 'cs.AI', 'cs.CV')
            max_papers: Maximum number of papers to fetch

        Returns:
            List of ArxivPaper objects
        """
        all_papers = []
        papers_per_request = min(self.max_results_per_request, max_papers)

        # Construct search query for category
        query = f"cat:{category}"

        start = 0
        while len(all_papers) < max_papers:
            remaining = max_papers - len(all_papers)
            batch_size = min(papers_per_request, remaining)

            papers = self.search_papers(query, max_results=batch_size, start=start)

            if not papers:  # No more papers available
                break

            all_papers.extend(papers)
            start += len(papers)

            print(f"Fetched {len(all_papers)}/{max_papers} papers from {category}")

            # If we got fewer papers than requested, we've reached the end
            if len(papers) < batch_size:
                break

        return all_papers

    def fetch_recent_papers(
        self, categories: List[str], max_papers_per_category: int = 1000
    ) -> Dict[str, List[ArxivPaper]]:
        """
        Fetch recent papers from multiple categories

        Args:
            categories: List of arXiv categories
            max_papers_per_category: Maximum papers per category

        Returns:
            Dictionary mapping category to list of papers
        """
        results = {}

        for category in categories:
            print(f"\n📚 Fetching papers from category: {category}")
            papers = self.fetch_papers_by_category(category, max_papers_per_category)
            results[category] = papers
            print(f"✅ Fetched {len(papers)} papers from {category}")

        return results


# Convenience functions
def create_arxiv_client() -> ArxivClient:
    """Create and return an ArxivClient instance"""
    return ArxivClient()


def fetch_papers_for_categories(
    categories: List[str] = None,
) -> Dict[str, List[ArxivPaper]]:
    """
    Fetch papers for the configured categories

    Args:
        categories: List of categories to fetch. If None, uses config.ARXIV_CATEGORIES

    Returns:
        Dictionary mapping category to list of papers
    """
    if categories is None:
        categories = config.ARXIV_CATEGORIES

    client = create_arxiv_client()
    return client.fetch_recent_papers(categories, config.MAX_PAPERS_PER_CATEGORY)
