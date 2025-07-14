#!/usr/bin/env python3
"""
Vector database implementation using FAISS for fast similarity search
"""

import numpy as np
import faiss
import pickle
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import time

from simple_embeddings import load_simple_embeddings, create_simple_embedding_generator
from data_storage import DataStorage, load_all_stored_papers
from arxiv_client import ArxivPaper
from config import config


class VectorDatabase:
    """FAISS-based vector database for semantic similarity search"""

    def __init__(self, embedding_dim: int = None):
        self.embedding_dim = embedding_dim
        self.index = None
        self.paper_ids = []
        self.paper_metadata = {}
        self.is_built = False

        # Storage
        self.storage = DataStorage()
        self.embeddings_dir = self.storage.embeddings_dir

        print(f"🗄️ Initializing VectorDatabase")
        if embedding_dim:
            print(f"📏 Embedding dimension: {embedding_dim}")

    def _create_index(
        self, embedding_dim: int, index_type: str = "flat"
    ) -> faiss.Index:
        """
        Create a FAISS index

        Args:
            embedding_dim: Dimension of embeddings
            index_type: Type of index ('flat', 'ivf', 'hnsw')

        Returns:
            FAISS index
        """
        if index_type == "flat":
            # Simple brute-force search (good for small datasets)
            index = faiss.IndexFlatIP(
                embedding_dim
            )  # Inner product (cosine similarity)
        elif index_type == "ivf":
            # Inverted file index (faster for larger datasets)
            quantizer = faiss.IndexFlatIP(embedding_dim)
            index = faiss.IndexIVFFlat(quantizer, embedding_dim, 100)  # 100 clusters
        elif index_type == "hnsw":
            # Hierarchical NSW (fast and accurate)
            index = faiss.IndexHNSWFlat(embedding_dim, 32)
        else:
            raise ValueError(f"Unknown index type: {index_type}")

        print(f"📊 Created {index_type} index with dimension {embedding_dim}")
        return index

    def build_index(
        self,
        embeddings: np.ndarray,
        paper_ids: List[str],
        papers: List[ArxivPaper] = None,
        index_type: str = "flat",
    ) -> None:
        """
        Build the vector index from embeddings

        Args:
            embeddings: numpy array of shape (n_papers, embedding_dim)
            paper_ids: list of paper IDs corresponding to embeddings
            papers: optional list of ArxivPaper objects for metadata
            index_type: type of FAISS index to use
        """
        if len(embeddings) != len(paper_ids):
            raise ValueError("Number of embeddings must match number of paper IDs")

        if len(embeddings) == 0:
            raise ValueError("Cannot build index with empty embeddings")

        print(f"🔨 Building vector index...")
        print(f"📊 Embeddings shape: {embeddings.shape}")
        print(f"📄 Number of papers: {len(paper_ids)}")

        # Store dimensions
        self.embedding_dim = embeddings.shape[1]

        # Normalize embeddings for cosine similarity
        embeddings_normalized = embeddings.copy().astype(np.float32)
        norms = np.linalg.norm(embeddings_normalized, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        embeddings_normalized = embeddings_normalized / norms

        # Create index
        self.index = self._create_index(self.embedding_dim, index_type)

        # Add embeddings to index
        self.index.add(embeddings_normalized)

        # Store metadata
        self.paper_ids = paper_ids.copy()

        # Store paper metadata if provided
        if papers:
            for paper in papers:
                if paper.id in paper_ids:
                    self.paper_metadata[paper.id] = {
                        "title": paper.title,
                        "authors": paper.authors,
                        "abstract": paper.abstract,
                        "published": paper.published,
                        "url": paper.url,
                        "categories": paper.categories,
                    }

        self.is_built = True
        print(f"✅ Vector index built successfully!")
        print(f"🔍 Index contains {self.index.ntotal} vectors")

    def search(
        self, query_embedding: np.ndarray, k: int = 10
    ) -> Tuple[List[str], List[float]]:
        """
        Search for similar papers using a query embedding

        Args:
            query_embedding: numpy array of shape (embedding_dim,)
            k: number of top results to return

        Returns:
            Tuple of (paper_ids, similarity_scores)
        """
        if not self.is_built:
            raise ValueError("Index must be built before searching")

        # Normalize query embedding
        query_normalized = query_embedding.copy().astype(np.float32).reshape(1, -1)
        norm = np.linalg.norm(query_normalized)
        if norm > 0:
            query_normalized = query_normalized / norm

        # Search
        scores, indices = self.index.search(
            query_normalized, min(k, len(self.paper_ids))
        )

        # Convert to paper IDs and scores
        paper_ids = [
            self.paper_ids[idx] for idx in indices[0] if idx < len(self.paper_ids)
        ]
        similarity_scores = scores[0].tolist()

        return paper_ids, similarity_scores

    def search_by_text(
        self, query_text: str, embedding_generator, k: int = 10
    ) -> List[Dict]:
        """
        Search for similar papers using a text query

        Args:
            query_text: text query
            embedding_generator: fitted embedding generator
            k: number of top results to return

        Returns:
            List of paper results with metadata and scores
        """
        if not self.is_built:
            raise ValueError("Index must be built before searching")

        # Generate query embedding
        query_embedding = embedding_generator.embed_query(query_text)

        # Search
        paper_ids, scores = self.search(query_embedding, k)

        # Combine with metadata
        results = []
        for paper_id, score in zip(paper_ids, scores):
            result = {
                "paper_id": paper_id,
                "similarity_score": float(score),
                "metadata": self.paper_metadata.get(paper_id, {}),
            }
            results.append(result)

        return results

    def save_index(self, filename: str = "vector_index") -> str:
        """
        Save the vector index to disk

        Args:
            filename: base filename (without extension)

        Returns:
            Path to saved file
        """
        if not self.is_built:
            raise ValueError("Cannot save unbuilt index")

        # Save FAISS index
        index_path = self.embeddings_dir / f"{filename}.faiss"
        faiss.write_index(self.index, str(index_path))

        # Save metadata
        metadata_path = self.embeddings_dir / f"{filename}_metadata.pkl"
        metadata = {
            "paper_ids": self.paper_ids,
            "paper_metadata": self.paper_metadata,
            "embedding_dim": self.embedding_dim,
            "created_at": time.time(),
            "index_type": type(self.index).__name__,
        }

        with open(metadata_path, "wb") as f:
            pickle.dump(metadata, f)

        print(f"💾 Saved vector index to {index_path}")
        print(f"💾 Saved metadata to {metadata_path}")

        return str(index_path)

    def load_index(self, filename: str = "vector_index") -> None:
        """
        Load a vector index from disk

        Args:
            filename: base filename (without extension)
        """
        # Load FAISS index
        index_path = self.embeddings_dir / f"{filename}.faiss"
        if not index_path.exists():
            raise FileNotFoundError(f"Index file not found: {index_path}")

        self.index = faiss.read_index(str(index_path))

        # Load metadata
        metadata_path = self.embeddings_dir / f"{filename}_metadata.pkl"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)

        self.paper_ids = metadata["paper_ids"]
        self.paper_metadata = metadata["paper_metadata"]
        self.embedding_dim = metadata["embedding_dim"]

        self.is_built = True

        print(f"📖 Loaded vector index from {index_path}")
        print(f"📊 Index type: {metadata.get('index_type', 'unknown')}")
        print(f"🔍 Index contains {self.index.ntotal} vectors")
        print(f"📏 Embedding dimension: {self.embedding_dim}")

    def get_stats(self) -> Dict:
        """
        Get statistics about the vector database

        Returns:
            Dictionary with database statistics
        """
        stats = {
            "is_built": self.is_built,
            "embedding_dim": self.embedding_dim,
            "num_papers": len(self.paper_ids) if self.is_built else 0,
            "index_size": self.index.ntotal if self.is_built else 0,
            "has_metadata": len(self.paper_metadata) > 0 if self.is_built else False,
        }

        if self.is_built:
            stats["index_type"] = type(self.index).__name__

        return stats


def build_vector_database_from_embeddings(
    embeddings_filename: str = "simple_paper_embeddings",
    index_filename: str = "vector_index",
) -> VectorDatabase:
    """
    Build a vector database from saved embeddings

    Args:
        embeddings_filename: filename of saved embeddings
        index_filename: filename for saving the vector index

    Returns:
        Built VectorDatabase instance
    """
    print("🚀 Building vector database from embeddings...")

    # Load embeddings
    embeddings, metadata = load_simple_embeddings(embeddings_filename)
    paper_ids = metadata["paper_ids"]

    # Load paper metadata
    papers = load_all_stored_papers()

    # Create and build database
    vector_db = VectorDatabase()
    vector_db.build_index(embeddings, paper_ids, papers)

    # Save index
    vector_db.save_index(index_filename)

    return vector_db


def load_vector_database(index_filename: str = "vector_index") -> VectorDatabase:
    """
    Load a vector database from disk

    Args:
        index_filename: filename of saved vector index

    Returns:
        Loaded VectorDatabase instance
    """
    vector_db = VectorDatabase()
    vector_db.load_index(index_filename)
    return vector_db


def search_papers(
    query_text: str,
    k: int = 10,
    index_filename: str = "vector_index",
    model_filename: str = "simple_embedding_model",
) -> List[Dict]:
    """
    Search for papers using a text query

    Args:
        query_text: text query
        k: number of results to return
        index_filename: filename of vector index
        model_filename: filename of embedding model

    Returns:
        List of search results
    """
    # Load vector database
    vector_db = load_vector_database(index_filename)

    # Load embedding generator
    embedding_generator = create_simple_embedding_generator()
    embedding_generator.load_model(model_filename)

    # Search
    results = vector_db.search_by_text(query_text, embedding_generator, k)

    return results
