#!/usr/bin/env python3
"""
Simple embedding generation using TF-IDF vectorization
This is a fallback approach to get the pipeline working while we resolve sentence-transformers dependencies
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from typing import List, Dict, Tuple, Optional
import pickle
import os
from pathlib import Path
import time

from arxiv_client import ArxivPaper
from data_storage import DataStorage, load_all_stored_papers
from config import config


class SimpleEmbeddingGenerator:
    """Generate embeddings using TF-IDF + SVD for paper abstracts"""

    def __init__(self, embedding_dim: int = 384, max_features: int = 10000):
        self.embedding_dim = embedding_dim
        self.max_features = max_features
        self.vectorizer = None
        self.svd = None
        self.is_fitted = False

        # Storage paths
        self.storage = DataStorage()
        self.embeddings_dir = self.storage.embeddings_dir

        print(f"🤖 Initializing SimpleEmbeddingGenerator")
        print(f"📊 Embedding dimension: {self.embedding_dim}")
        print(f"📚 Max features: {self.max_features}")

    def _preprocess_text(self, text: str) -> str:
        """Simple text preprocessing"""
        if not text:
            return ""

        # Convert to lowercase and basic cleaning
        text = text.lower()
        # Keep only alphanumeric characters, spaces, and basic punctuation
        import re

        text = re.sub(r"[^a-zA-Z0-9\s\-.,;:!?()]", " ", text)
        text = re.sub(r"\s+", " ", text.strip())

        return text

    def fit(self, texts: List[str]) -> None:
        """
        Fit the TF-IDF vectorizer and SVD on the corpus

        Args:
            texts: List of text strings to fit on
        """
        if not texts:
            raise ValueError("Cannot fit on empty text list")

        print(f"🔄 Fitting TF-IDF vectorizer on {len(texts)} texts...")

        # Preprocess texts
        processed_texts = [self._preprocess_text(text) for text in texts]

        # Initialize TF-IDF vectorizer
        # Adjust parameters based on corpus size
        min_df = max(1, min(2, len(processed_texts) // 10))
        max_df = min(0.95, 1.0 - (1.0 / len(processed_texts)))

        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            stop_words="english",
            ngram_range=(1, 2),  # Include unigrams and bigrams
            min_df=min_df,  # Adaptive min_df
            max_df=max_df,  # Adaptive max_df
            sublinear_tf=True,
        )

        # Fit and transform texts
        tfidf_matrix = self.vectorizer.fit_transform(processed_texts)
        print(f"📊 TF-IDF matrix shape: {tfidf_matrix.shape}")

        # Initialize SVD for dimensionality reduction
        # Adjust n_components to be at most the number of features
        n_features = tfidf_matrix.shape[1]
        n_components = min(self.embedding_dim, n_features - 1) if n_features > 1 else 1

        self.svd = TruncatedSVD(n_components=n_components, random_state=42)

        # Fit SVD on TF-IDF matrix
        self.svd.fit(tfidf_matrix)

        explained_variance = self.svd.explained_variance_ratio_.sum()
        print(f"📈 SVD explained variance: {explained_variance:.3f}")

        self.is_fitted = True
        print("✅ Model fitted successfully!")

    def transform(self, texts: List[str]) -> np.ndarray:
        """
        Transform texts to embeddings

        Args:
            texts: List of text strings to embed

        Returns:
            numpy array of shape (len(texts), embedding_dim)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before transform. Call fit() first.")

        if not texts:
            return np.array([])

        # Preprocess texts
        processed_texts = [self._preprocess_text(text) for text in texts]

        # Transform to TF-IDF
        tfidf_matrix = self.vectorizer.transform(processed_texts)

        # Transform to embeddings using SVD
        embeddings = self.svd.transform(tfidf_matrix)

        return embeddings

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """
        Fit the model and transform texts to embeddings

        Args:
            texts: List of text strings

        Returns:
            numpy array of shape (len(texts), embedding_dim)
        """
        self.fit(texts)
        return self.transform(texts)

    def generate_paper_embeddings(
        self, papers: List[ArxivPaper]
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Generate embeddings for paper abstracts

        Args:
            papers: List of ArxivPaper objects

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
        embeddings = self.fit_transform(abstracts)

        print(f"✅ Generated embeddings shape: {embeddings.shape}")
        return embeddings, paper_ids

    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a single query string

        Args:
            query: Query text to embed

        Returns:
            numpy array of shape (embedding_dim,)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before embedding queries")

        embeddings = self.transform([query])
        return embeddings[0]  # Return single embedding vector

    def save_model(self, filename: str = "simple_embedding_model") -> str:
        """
        Save the fitted model to disk

        Args:
            filename: base filename (without extension)

        Returns:
            Path to saved file
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")

        filepath = self.embeddings_dir / f"{filename}.pkl"

        # Create model data
        model_data = {
            "vectorizer": self.vectorizer,
            "svd": self.svd,
            "embedding_dim": self.embedding_dim,
            "max_features": self.max_features,
            "is_fitted": self.is_fitted,
            "created_at": time.time(),
            "model_type": "tfidf_svd",
        }

        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)

        print(f"💾 Saved model to {filepath}")
        return str(filepath)

    def load_model(self, filename: str = "simple_embedding_model") -> None:
        """
        Load a fitted model from disk

        Args:
            filename: base filename (without extension)
        """
        filepath = self.embeddings_dir / f"{filename}.pkl"

        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")

        try:
            with open(filepath, "rb") as f:
                model_data = pickle.load(f)

            self.vectorizer = model_data["vectorizer"]
            self.svd = model_data["svd"]
            self.embedding_dim = model_data["embedding_dim"]
            self.max_features = model_data["max_features"]
            self.is_fitted = model_data["is_fitted"]

            print(f"📖 Loaded model from {filepath}")
            print(f"📊 Model type: {model_data.get('model_type', 'unknown')}")
            print(f"📏 Embedding dimension: {self.embedding_dim}")

        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise

    def save_embeddings(
        self,
        embeddings: np.ndarray,
        paper_ids: List[str],
        filename: str = "simple_paper_embeddings",
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
            "model_type": "tfidf_svd",
            "embedding_dimension": self.embedding_dim,
            "max_features": self.max_features,
            "num_embeddings": len(embeddings),
            "paper_ids": paper_ids,
            "created_at": time.time(),
        }

        # Save data
        data = {"embeddings": embeddings, "metadata": metadata}

        with open(filepath, "wb") as f:
            pickle.dump(data, f)

        print(f"💾 Saved {len(embeddings)} embeddings to {filepath}")
        print(f"📊 Embedding shape: {embeddings.shape}")
        return str(filepath)

    def process_all_papers(self, filename: str = "simple_paper_embeddings") -> str:
        """
        Process all stored papers and generate embeddings

        Args:
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
        embeddings, paper_ids = self.generate_paper_embeddings(papers)

        if len(embeddings) == 0:
            raise ValueError("No valid abstracts found for embedding generation")

        # Save model
        model_path = self.save_model()

        # Save embeddings
        embeddings_path = self.save_embeddings(embeddings, paper_ids, filename)

        print(f"🎉 Successfully processed {len(paper_ids)} papers!")
        print(f"📁 Model saved to: {model_path}")
        print(f"📁 Embeddings saved to: {embeddings_path}")

        return embeddings_path


# Convenience functions
def create_simple_embedding_generator(
    embedding_dim: int = 384,
) -> SimpleEmbeddingGenerator:
    """Create and return a SimpleEmbeddingGenerator instance"""
    return SimpleEmbeddingGenerator(embedding_dim=embedding_dim)


def generate_simple_embeddings_for_papers(embedding_dim: int = 384) -> str:
    """
    Generate simple embeddings for all stored papers

    Args:
        embedding_dim: Embedding dimension

    Returns:
        Path to saved embeddings file
    """
    generator = create_simple_embedding_generator(embedding_dim)
    return generator.process_all_papers()


def load_simple_embeddings(
    filename: str = "simple_paper_embeddings",
) -> Tuple[np.ndarray, Dict]:
    """
    Load simple embeddings from disk

    Args:
        filename: Filename without extension

    Returns:
        Tuple of (embeddings, metadata)
    """
    storage = DataStorage()
    filepath = storage.embeddings_dir / f"{filename}.pkl"

    if not filepath.exists():
        raise FileNotFoundError(f"Embeddings file not found: {filepath}")

    with open(filepath, "rb") as f:
        data = pickle.load(f)

    return data["embeddings"], data["metadata"]
