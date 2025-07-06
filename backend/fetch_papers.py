#!/usr/bin/env python3
"""
CLI script to fetch papers from arXiv and store them locally
"""

import argparse
import sys
from typing import List
from arxiv_client import fetch_papers_for_categories, create_arxiv_client
from data_storage import save_fetched_papers, create_data_storage
from config import config


def main():
    parser = argparse.ArgumentParser(description="Fetch academic papers from arXiv")

    parser.add_argument(
        "--categories",
        nargs="+",
        default=config.ARXIV_CATEGORIES,
        help=f"arXiv categories to fetch (default: {config.ARXIV_CATEGORIES})",
    )

    parser.add_argument(
        "--max-papers",
        type=int,
        default=config.MAX_PAPERS_PER_CATEGORY,
        help=f"Maximum papers per category (default: {config.MAX_PAPERS_PER_CATEGORY})",
    )

    parser.add_argument(
        "--data-dir",
        default=config.DATA_DIR,
        help=f"Data directory (default: {config.DATA_DIR})",
    )

    parser.add_argument(
        "--test",
        action="store_true",
        help="Run a quick test with fewer papers (10 per category)",
    )

    parser.add_argument(
        "--stats", action="store_true", help="Show statistics about stored papers"
    )

    parser.add_argument(
        "--search",
        type=str,
        help="Search for specific papers (e.g., 'attention mechanism')",
    )

    args = parser.parse_args()

    # Show statistics if requested
    if args.stats:
        show_stats(args.data_dir)
        return

    # Search for specific papers if requested
    if args.search:
        search_papers(args.search, args.max_papers)
        return

    # Set test parameters
    if args.test:
        args.max_papers = 10
        print("🧪 Running in test mode (10 papers per category)")

    print("🚀 Starting arXiv paper fetching...")
    print(f"📂 Data directory: {args.data_dir}")
    print(f"📚 Categories: {args.categories}")
    print(f"📄 Max papers per category: {args.max_papers}")
    print("-" * 50)

    try:
        # Fetch papers
        papers_by_category = fetch_papers_for_categories(args.categories)

        # Limit papers if needed
        if args.max_papers != config.MAX_PAPERS_PER_CATEGORY:
            for category in papers_by_category:
                papers_by_category[category] = papers_by_category[category][
                    : args.max_papers
                ]

        # Show summary
        total_papers = sum(len(papers) for papers in papers_by_category.values())
        print(f"\n📊 FETCH SUMMARY:")
        print(f"Total papers fetched: {total_papers}")
        for category, papers in papers_by_category.items():
            print(f"  {category}: {len(papers)} papers")

        if total_papers == 0:
            print(
                "❌ No papers were fetched. Check your internet connection or try different categories."
            )
            sys.exit(1)

        # Save papers
        print("\n💾 Saving papers to storage...")
        saved_files = save_fetched_papers(papers_by_category, args.data_dir)

        print(f"\n✅ STORAGE SUMMARY:")
        for category, filepath in saved_files.items():
            print(f"  {category}: {filepath}")

        # Show sample papers
        print(f"\n📖 SAMPLE PAPERS:")
        sample_count = 0
        for category, papers in papers_by_category.items():
            if papers and sample_count < 3:
                paper = papers[0]
                print(f"\n  Category: {category}")
                print(f"  Title: {paper.title[:100]}...")
                print(f"  Authors: {', '.join(paper.authors[:3])}...")
                print(f"  Published: {paper.published[:10]}")
                sample_count += 1

        print(f"\n🎉 Successfully fetched and stored {total_papers} papers!")

    except Exception as e:
        print(f"❌ Error during paper fetching: {e}")
        sys.exit(1)


def show_stats(data_dir: str):
    """Show statistics about stored papers"""
    storage = create_data_storage(data_dir)
    stats = storage.get_papers_stats()

    print("📊 STORAGE STATISTICS")
    print("-" * 30)
    print(f"Total papers: {stats['total_papers']}")
    print(f"Categories: {len(stats['categories'])}")
    print(f"Raw files: {len(stats['raw_files'])}")
    print(f"Processed files: {len(stats['processed_files'])}")

    if stats["categories"]:
        print(f"\nCategories found:")
        for category in sorted(stats["categories"]):
            print(f"  - {category}")

    if stats["raw_files"]:
        print(f"\nRaw data files:")
        for filename in stats["raw_files"]:
            print(f"  - {filename}")


def search_papers(query: str, max_results: int = 50):
    """Search for specific papers"""
    print(f"🔍 Searching for: '{query}'")
    print(f"📄 Max results: {max_results}")
    print("-" * 40)

    try:
        client = create_arxiv_client()
        papers = client.search_papers(query, max_results=max_results)

        if not papers:
            print("❌ No papers found for this query.")
            return

        print(f"✅ Found {len(papers)} papers:")
        for i, paper in enumerate(papers[:10], 1):  # Show first 10
            print(f"\n{i}. {paper.title}")
            print(
                f"   Authors: {', '.join(paper.authors[:3])}{'...' if len(paper.authors) > 3 else ''}"
            )
            print(f"   Published: {paper.published[:10]}")
            print(f"   Categories: {', '.join(paper.categories)}")
            print(f"   URL: {paper.url}")

        if len(papers) > 10:
            print(f"\n... and {len(papers) - 10} more papers")

    except Exception as e:
        print(f"❌ Error during search: {e}")


if __name__ == "__main__":
    main()
