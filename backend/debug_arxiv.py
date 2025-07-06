#!/usr/bin/env python3
"""
Debug script to test arXiv API connection
"""

import requests
import xml.etree.ElementTree as ET
from config import config


def test_arxiv_api():
    """Test the arXiv API with a simple request"""

    # Try a simple search
    base_url = config.ARXIV_API_BASE_URL
    params = {"search_query": "all:neural networks", "start": 0, "max_results": 3}

    url = f"{base_url}/query"

    print(f"🔗 Testing arXiv API: {url}")
    print(f"📋 Parameters: {params}")
    print("-" * 50)

    try:
        response = requests.get(url, params=params, timeout=30)
        print(f"📡 Response status: {response.status_code}")
        print(f"📏 Response length: {len(response.content)} bytes")

        if response.status_code != 200:
            print(f"❌ HTTP Error: {response.status_code}")
            print(f"Response content: {response.text[:500]}")
            return

        # Try to parse the XML
        try:
            root = ET.fromstring(response.content)
            print(f"✅ XML parsing successful")

            # Print the XML structure
            print(f"🏷️ Root tag: {root.tag}")

            # Look for entries
            entries = root.findall("{http://www.w3.org/2005/Atom}entry")
            print(f"📄 Found {len(entries)} entries")

            if entries:
                # Print details of the first entry
                entry = entries[0]
                print(f"\n📖 First entry details:")

                id_elem = entry.find("{http://www.w3.org/2005/Atom}id")
                title_elem = entry.find("{http://www.w3.org/2005/Atom}title")
                summary_elem = entry.find("{http://www.w3.org/2005/Atom}summary")

                if id_elem is not None:
                    print(f"  ID: {id_elem.text}")
                if title_elem is not None:
                    print(f"  Title: {title_elem.text[:100]}...")
                if summary_elem is not None:
                    print(f"  Abstract: {summary_elem.text[:200]}...")
            else:
                # Print the raw XML to see what we're getting
                print(f"\n📄 Raw XML response (first 1000 chars):")
                print(response.text[:1000])

        except ET.ParseError as e:
            print(f"❌ XML parsing failed: {e}")
            print(f"Raw response (first 500 chars): {response.text[:500]}")

    except requests.RequestException as e:
        print(f"❌ Request failed: {e}")


def test_category_search():
    """Test category-specific search"""

    base_url = config.ARXIV_API_BASE_URL
    params = {"search_query": "cat:cs.AI", "start": 0, "max_results": 3}

    url = f"{base_url}/query"

    print(f"\n🎯 Testing category search: cs.AI")
    print(f"📋 Parameters: {params}")
    print("-" * 50)

    try:
        response = requests.get(url, params=params, timeout=30)
        print(f"📡 Response status: {response.status_code}")

        if response.status_code == 200:
            root = ET.fromstring(response.content)
            entries = root.findall("{http://www.w3.org/2005/Atom}entry")
            print(f"📄 Found {len(entries)} entries in cs.AI category")

            if not entries:
                print(f"Raw response (first 500 chars): {response.text[:500]}")

    except Exception as e:
        print(f"❌ Category search failed: {e}")


if __name__ == "__main__":
    print("🔍 Debugging arXiv API connection...")
    test_arxiv_api()
    test_category_search()
    print("\n✅ Debug complete!")
