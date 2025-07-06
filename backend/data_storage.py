import json
import os
from typing import List, Dict, Optional
import pandas as pd
from datetime import datetime
from pathlib import Path
from arxiv_client import ArxivPaper
from config import config


class DataStorage:
    """Handle storage and retrieval of arXiv paper data"""

    def __init__(self, data_dir: str = None):
        self.data_dir = Path(data_dir or config.DATA_DIR)
        self.data_dir.mkdir(exist_ok=True)

        # Create subdirectories
        self.raw_data_dir = self.data_dir / "raw"
        self.processed_data_dir = self.data_dir / "processed"
        self.embeddings_dir = self.data_dir / "embeddings"

        for dir_path in [
            self.raw_data_dir,
            self.processed_data_dir,
            self.embeddings_dir,
        ]:
            dir_path.mkdir(exist_ok=True)

    def save_papers_json(self, papers: List[ArxivPaper], filename: str) -> str:
        """
        Save papers to JSON file

        Args:
            papers: List of ArxivPaper objects
            filename: Name of the file (without extension)

        Returns:
            Path to the saved file
        """
        filepath = self.raw_data_dir / f"{filename}.json"

        # Convert papers to dictionaries
        papers_data = [paper.to_dict() for paper in papers]

        # Add metadata
        data = {
            "metadata": {
                "total_papers": len(papers),
                "created_at": datetime.now().isoformat(),
                "filename": filename,
            },
            "papers": papers_data,
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"💾 Saved {len(papers)} papers to {filepath}")
        return str(filepath)

    def load_papers_json(self, filename: str) -> List[ArxivPaper]:
        """
        Load papers from JSON file

        Args:
            filename: Name of the file (without extension)

        Returns:
            List of ArxivPaper objects
        """
        filepath = self.raw_data_dir / f"{filename}.json"

        if not filepath.exists():
            print(f"❌ File not found: {filepath}")
            return []

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            papers = []
            for paper_dict in data.get("papers", []):
                paper = ArxivPaper(**paper_dict)
                papers.append(paper)

            print(f"📖 Loaded {len(papers)} papers from {filepath}")
            return papers

        except Exception as e:
            print(f"❌ Error loading papers from {filepath}: {e}")
            return []

    def save_papers_by_category(
        self, papers_by_category: Dict[str, List[ArxivPaper]]
    ) -> Dict[str, str]:
        """
        Save papers organized by category

        Args:
            papers_by_category: Dictionary mapping category to list of papers

        Returns:
            Dictionary mapping category to saved file path
        """
        saved_files = {}

        for category, papers in papers_by_category.items():
            if papers:
                filename = f"papers_{category.replace('.', '_')}"
                filepath = self.save_papers_json(papers, filename)
                saved_files[category] = filepath

        # Save a combined file with all papers
        all_papers = []
        for papers in papers_by_category.values():
            all_papers.extend(papers)

        if all_papers:
            combined_filepath = self.save_papers_json(all_papers, "all_papers")
            saved_files["combined"] = combined_filepath

        return saved_files

    def load_all_papers(self) -> List[ArxivPaper]:
        """
        Load all papers from the combined file

        Returns:
            List of all ArxivPaper objects
        """
        return self.load_papers_json("all_papers")

    def save_papers_dataframe(self, papers: List[ArxivPaper], filename: str) -> str:
        """
        Save papers as a pandas DataFrame (CSV format)

        Args:
            papers: List of ArxivPaper objects
            filename: Name of the file (without extension)

        Returns:
            Path to the saved CSV file
        """
        filepath = self.processed_data_dir / f"{filename}.csv"

        # Convert to DataFrame
        papers_data = []
        for paper in papers:
            data = paper.to_dict()
            # Convert lists to strings for CSV compatibility
            data["authors"] = "; ".join(data["authors"])
            data["categories"] = "; ".join(data["categories"])
            papers_data.append(data)

        df = pd.DataFrame(papers_data)
        df.to_csv(filepath, index=False, encoding="utf-8")

        print(f"📊 Saved {len(papers)} papers to CSV: {filepath}")
        return str(filepath)

    def load_papers_dataframe(self, filename: str) -> pd.DataFrame:
        """
        Load papers as a pandas DataFrame

        Args:
            filename: Name of the file (without extension)

        Returns:
            DataFrame with paper data
        """
        filepath = self.processed_data_dir / f"{filename}.csv"

        if not filepath.exists():
            print(f"❌ File not found: {filepath}")
            return pd.DataFrame()

        try:
            df = pd.read_csv(filepath, encoding="utf-8")
            print(f"📊 Loaded DataFrame with {len(df)} papers from {filepath}")
            return df
        except Exception as e:
            print(f"❌ Error loading DataFrame from {filepath}: {e}")
            return pd.DataFrame()

    def get_papers_stats(self) -> Dict:
        """
        Get statistics about stored papers

        Returns:
            Dictionary with storage statistics
        """
        stats = {
            "raw_files": [],
            "processed_files": [],
            "total_papers": 0,
            "categories": set(),
        }

        # Check raw data files
        for file_path in self.raw_data_dir.glob("*.json"):
            stats["raw_files"].append(file_path.name)

        # Check processed data files
        for file_path in self.processed_data_dir.glob("*.csv"):
            stats["processed_files"].append(file_path.name)

        # Try to get total papers count from combined file
        try:
            combined_path = self.raw_data_dir / "all_papers.json"
            if combined_path.exists():
                with open(combined_path, "r") as f:
                    data = json.load(f)
                    stats["total_papers"] = data.get("metadata", {}).get(
                        "total_papers", 0
                    )

                    # Extract categories
                    for paper in data.get("papers", []):
                        for category in paper.get("categories", []):
                            stats["categories"].add(category)
        except Exception as e:
            print(f"❌ Error reading stats: {e}")

        stats["categories"] = list(stats["categories"])
        return stats

    def cleanup_old_files(self, days_old: int = 7) -> None:
        """
        Remove files older than specified days

        Args:
            days_old: Remove files older than this many days
        """
        from datetime import datetime, timedelta

        cutoff_date = datetime.now() - timedelta(days=days_old)

        for directory in [
            self.raw_data_dir,
            self.processed_data_dir,
            self.embeddings_dir,
        ]:
            for file_path in directory.iterdir():
                if file_path.is_file():
                    file_modified = datetime.fromtimestamp(file_path.stat().st_mtime)
                    if file_modified < cutoff_date:
                        file_path.unlink()
                        print(f"🗑️ Removed old file: {file_path}")


# Convenience functions
def create_data_storage(data_dir: str = None) -> DataStorage:
    """Create and return a DataStorage instance"""
    return DataStorage(data_dir)


def save_fetched_papers(
    papers_by_category: Dict[str, List[ArxivPaper]], data_dir: str = None
) -> Dict[str, str]:
    """
    Save fetched papers to storage

    Args:
        papers_by_category: Dictionary mapping category to list of papers
        data_dir: Data directory path (optional)

    Returns:
        Dictionary mapping category to saved file path
    """
    storage = create_data_storage(data_dir)
    return storage.save_papers_by_category(papers_by_category)


def load_all_stored_papers(data_dir: str = None) -> List[ArxivPaper]:
    """
    Load all stored papers

    Args:
        data_dir: Data directory path (optional)

    Returns:
        List of all stored ArxivPaper objects
    """
    storage = create_data_storage(data_dir)
    return storage.load_all_papers()
