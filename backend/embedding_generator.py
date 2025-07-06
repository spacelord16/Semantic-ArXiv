#!/usr/bin/env python3
"""
Embedding generation module for converting paper abstracts to semantic vectors
"""

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple, Optional
import pickle
import os
from pathlib import Path
import time
from tqdm import tqdm

from arxiv_client import ArxivPaper
from data_storage import DataStorage, load_all_stored_papers
from config import config


class EmbeddingGenerator:
    """Generate semantic embeddings for paper abstracts using sentence transformers"""

    def __init__(self, model_name: str = None, device: str = None):
        self.model_name = model_name or config.EMBEDDING_MODEL
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.embedding_dim = config.EMBEDDING_DIMENSION

        # Storage paths
        self.storage = DataStorage()
        self.embeddings_dir = self.storage.embeddings_dir

        print(f"🤖 Initializing EmbeddingGenerator with model: {self.model_name}")
        print(f"💻 Using device: {self.device}")

    def load_model(self) -> None:
        """Load the sentence transformer model"""
        if self.model is None:
            print(f"📦 Loading model: {self.model_name}")
            try:
                self.model = SentenceTransformer(self.model_name, device=self.device)
                actual_dim = self.model.get_sentence_embedding_dimension()

                if actual_dim != self.embedding_dim:
                    print(
                        f"⚠️ Model dimension ({actual_dim}) differs from config ({self.embedding_dim})"
                    )
                    self.embedding_dim = actual_dim

                print(
                    f"✅ Model loaded successfully! Embedding dimension: {self.embedding_dim}"
                )
            except Exception as e:
                print(f"❌ Error loading model: {e}")
                raise

    def generate_embeddings(
        self, texts: List[str], batch_size: int = 32, show_progress: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for a list of texts

        Args:
            texts: List of text strings to embed
            batch_size: Batch size for processing
            show_progress: Whether to show progress bar

        Returns:
            numpy array of shape (len(texts), embedding_dim)
        """
        if not texts:
            return np.array([])

        if self.model is None:
            self.load_model()

        print(f"🔄 Generating embeddings for {len(texts)} texts...")

        try:
            # Generate embeddings with progress bar
            if show_progress:
                embeddings = self.model.encode(
                    texts,
                    batch_size=batch_size,
                    show_progress_bar=True,
                    convert_to_numpy=True,
                )
            else:
                embeddings = self.model.encode(
                    texts, batch_size=batch_size, convert_to_numpy=True
                )

            print(f"✅ Generated embeddings shape: {embeddings.shape}")
            return embeddings

        except Exception as e:
            print(f"❌ Error generating embeddings: {e}")
            raise

    def generate_paper_embeddings(
        self, papers: List[ArxivPaper], batch_size: int = 32
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Generate embeddings for paper abstracts

        Args:
            papers: List of ArxivPaper objects
            batch_size: Batch size for processing

        Returns:
            Tuple of (embeddings array, list of paper IDs)
        """
        if not papers:
            return np.array([]), []

        # Extract abstracts and IDs
        abstracts = []
        paper_ids = []

        for paper in papers:
            if paper.abstract and paper.abstract.strip():
                abstracts.append(paper.abstract)
                paper_ids.append(paper.id)

        print(f"📄 Processing {len(abstracts)} paper abstracts...")

        # Generate embeddings
        embeddings = self.generate_embeddings(abstracts, batch_size=batch_size)

        return embeddings, paper_ids

    def save_embeddings(
        self,
        embeddings: np.ndarray,
        paper_ids: List[str],
        filename: str = "paper_embeddings",
    ) -> str:
        """
        Save embeddings and metadata to disk

        Args:
            embeddings: numpy array of embeddings
            paper_ids: list of corresponding paper IDs
            filename: base filename (without extension)

        Returns:
            Path to saved file
        """
        if len(embeddings) != len(paper_ids):
            raise ValueError("Number of embeddings must match number of paper IDs")

        filepath = self.embeddings_dir / f"{filename}.pkl"

        # Create metadata
        metadata = {
            "model_name": self.model_name,
            "embedding_dimension": self.embedding_dim,
            "num_embeddings": len(embeddings),
            "paper_ids": paper_ids,
            "created_at": time.time(),
            "device": self.device,
        }

        # Save data
        data = {"embeddings": embeddings, "metadata": metadata}

        with open(filepath, "wb") as f:
            pickle.dump(data, f)

        print(f"💾 Saved {len(embeddings)} embeddings to {filepath}")
        print(f"📊 Embedding shape: {embeddings.shape}")
        return str(filepath)

    def load_embeddings(
        self, filename: str = "paper_embeddings"
    ) -> Tuple[np.ndarray, Dict]:
        """
        Load embeddings and metadata from disk

        Args:
            filename: base filename (without extension)

        Returns:
            Tuple of (embeddings array, metadata dict)
        """
        filepath = self.embeddings_dir / f"{filename}.pkl"

        if not filepath.exists():
            raise FileNotFoundError(f"Embeddings file not found: {filepath}")

        try:
            with open(filepath, "rb") as f:
                data = pickle.load(f)

            embeddings = data["embeddings"]
            metadata = data["metadata"]

            print(f"📖 Loaded {len(embeddings)} embeddings from {filepath}")
            print(f"📊 Embedding shape: {embeddings.shape}")
            print(f"🤖 Model: {metadata.get('model_name', 'unknown')}")

            return embeddings, metadata

        except Exception as e:
            print(f"❌ Error loading embeddings: {e}")
            raise

    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a single query string

        Args:
            query: Query text to embed

        Returns:
            numpy array of shape (embedding_dim,)
        """
        if self.model is None:
            self.load_model()

        embedding = self.model.encode([query], convert_to_numpy=True)
        return embedding[0]  # Return single embedding vector

    def process_all_papers(
        self, batch_size: int = 32, filename: str = "paper_embeddings"
    ) -> str:
        """
        Process all stored papers and generate embeddings

        Args:
            batch_size: Batch size for processing
            filename: Output filename

        Returns:
            Path to saved embeddings file
        """
        # Load all papers
        print("📚 Loading all stored papers...")
        papers = load_all_stored_papers()

        if not papers:
            raise ValueError("No papers found in storage. Please fetch papers first.")

        print(f"📄 Found {len(papers)} papers")

        # Generate embeddings
        embeddings, paper_ids = self.generate_paper_embeddings(papers, batch_size)

        if len(embeddings) == 0:
            raise ValueError("No valid abstracts found for embedding generation")

        # Save embeddings
        filepath = self.save_embeddings(embeddings, paper_ids, filename)

        print(f"🎉 Successfully processed {len(paper_ids)} papers!")
        return filepath

    def get_embedding_stats(self) -> Dict:
        """
        Get statistics about stored embeddings

        Returns:
            Dictionary with embedding statistics
        """
        stats = {
            "embedding_files": [],
            "total_embeddings": 0,
            "models_used": set(),
            "embedding_dimensions": set(),
        }

        # Check all pickle files in embeddings directory
        for file_path in self.embeddings_dir.glob("*.pkl"):
            try:
                with open(file_path, "rb") as f:
                    data = pickle.load(f)

                metadata = data.get("metadata", {})
                embeddings = data.get("embeddings", np.array([]))

                file_info = {
                    "filename": file_path.name,
                    "num_embeddings": len(embeddings),
                    "model_name": metadata.get("model_name", "unknown"),
                    "embedding_dimension": metadata.get(
                        "embedding_dimension", "unknown"
                    ),
                    "created_at": metadata.get("created_at", 0),
                }

                stats["embedding_files"].append(file_info)
                stats["total_embeddings"] += len(embeddings)
                stats["models_used"].add(metadata.get("model_name", "unknown"))
                stats["embedding_dimensions"].add(
                    metadata.get("embedding_dimension", "unknown")
                )

            except Exception as e:
                print(f"⚠️ Error reading {file_path}: {e}")

        stats["models_used"] = list(stats["models_used"])
        stats["embedding_dimensions"] = list(stats["embedding_dimensions"])

        return stats


# Convenience functions
def create_embedding_generator(model_name: str = None) -> EmbeddingGenerator:
    """Create and return an EmbeddingGenerator instance"""
    return EmbeddingGenerator(model_name)


def generate_embeddings_for_stored_papers(
    model_name: str = None, batch_size: int = 32
) -> str:
    """
    Generate embeddings for all stored papers

    Args:
        model_name: Sentence transformer model name
        batch_size: Batch size for processing

    Returns:
        Path to saved embeddings file
    """
    generator = create_embedding_generator(model_name)
    return generator.process_all_papers(batch_size)


def load_paper_embeddings(
    filename: str = "paper_embeddings",
) -> Tuple[np.ndarray, Dict]:
    """
    Load paper embeddings from disk

    Args:
        filename: Filename without extension

    Returns:
        Tuple of (embeddings, metadata)
    """
    generator = create_embedding_generator()
    return generator.load_embeddings(filename)
