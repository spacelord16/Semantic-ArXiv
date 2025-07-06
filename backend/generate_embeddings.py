#!/usr/bin/env python3
"""
CLI script to generate embeddings for stored papers
"""

import argparse
import sys
from embedding_generator import (
    create_embedding_generator,
    generate_embeddings_for_stored_papers,
)
from data_storage import create_data_storage
from config import config


def main():
    parser = argparse.ArgumentParser(
        description="Generate embeddings for stored papers"
    )

    parser.add_argument(
        "--model",
        default=config.EMBEDDING_MODEL,
        help=f"Sentence transformer model name (default: {config.EMBEDDING_MODEL})",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for processing (default: 32)",
    )

    parser.add_argument(
        "--filename",
        default="paper_embeddings",
        help="Output filename for embeddings (default: paper_embeddings)",
    )

    parser.add_argument(
        "--stats", action="store_true", help="Show embedding statistics"
    )

    parser.add_argument(
        "--test-query", type=str, help="Test embedding generation with a query"
    )

    args = parser.parse_args()

    # Show statistics if requested
    if args.stats:
        show_embedding_stats()
        return

    # Test query embedding if requested
    if args.test_query:
        test_query_embedding(args.test_query, args.model)
        return

    print("🚀 Starting embedding generation...")
    print(f"🤖 Model: {args.model}")
    print(f"📦 Batch size: {args.batch_size}")
    print(f"💾 Output file: {args.filename}")
    print("-" * 50)

    try:
        # Generate embeddings for all stored papers
        filepath = generate_embeddings_for_stored_papers(
            model_name=args.model, batch_size=args.batch_size
        )

        print(f"\n✅ SUCCESS!")
        print(f"📁 Embeddings saved to: {filepath}")

        # Show final statistics
        print(f"\n📊 FINAL STATISTICS:")
        show_embedding_stats()

    except Exception as e:
        print(f"❌ Error during embedding generation: {e}")
        sys.exit(1)


def show_embedding_stats():
    """Show statistics about stored embeddings"""
    generator = create_embedding_generator()
    stats = generator.get_embedding_stats()

    print("📊 EMBEDDING STATISTICS")
    print("-" * 30)
    print(f"Total embeddings: {stats['total_embeddings']}")
    print(f"Embedding files: {len(stats['embedding_files'])}")
    print(f"Models used: {stats['models_used']}")
    print(f"Dimensions: {stats['embedding_dimensions']}")

    if stats["embedding_files"]:
        print(f"\nEmbedding files:")
        for file_info in stats["embedding_files"]:
            print(f"  📄 {file_info['filename']}")
            print(f"     Embeddings: {file_info['num_embeddings']}")
            print(f"     Model: {file_info['model_name']}")
            print(f"     Dimension: {file_info['embedding_dimension']}")


def test_query_embedding(query: str, model_name: str):
    """Test embedding generation with a query"""
    print(f"🔍 Testing query embedding: '{query}'")
    print(f"🤖 Model: {model_name}")
    print("-" * 40)

    try:
        generator = create_embedding_generator(model_name)
        embedding = generator.embed_query(query)

        print(f"✅ Query embedded successfully!")
        print(f"📊 Embedding shape: {embedding.shape}")
        print(f"📈 Embedding stats:")
        print(f"   Mean: {embedding.mean():.6f}")
        print(f"   Std:  {embedding.std():.6f}")
        print(f"   Min:  {embedding.min():.6f}")
        print(f"   Max:  {embedding.max():.6f}")

        # Show first few values
        print(f"📝 First 10 values: {embedding[:10]}")

    except Exception as e:
        print(f"❌ Error during query embedding: {e}")


if __name__ == "__main__":
    main()
