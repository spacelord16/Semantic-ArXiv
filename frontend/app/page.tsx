"use client";

import { useState } from "react";
import { Search, BookOpen, Brain, Sparkles } from "lucide-react";
import axios from "axios";
import PaperCard from "./components/PaperCard";

// TypeScript interfaces for our API
interface Paper {
  id: string;
  title: string;
  authors: string[];
  abstract: string;
  published: string;
  url: string;
  similarity_score: number;
}

interface SearchResponse {
  papers: Paper[];
  query: string;
  total_results: number;
}

export default function Home() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<Paper[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [hasSearched, setHasSearched] = useState(false);

  const API_BASE_URL =
    process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

  const handleSearch = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();

    if (!query.trim()) {
      setError("Please enter a search query");
      return;
    }

    setLoading(true);
    setError("");
    setHasSearched(true);

    try {
      const response = await axios.post<SearchResponse>(
        `${API_BASE_URL}/search`,
        {
          query: query.trim(),
          limit: 10,
        }
      );

      setResults(response.data.papers);
      setError("");
    } catch (err) {
      console.error("Search error:", err);
      setError("Failed to search papers. Please try again.");
      setResults([]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      <div className="container mx-auto px-4 py-12">
        {/* Hero Section */}
        <div className="text-center mb-12">
          <div className="flex justify-center mb-4">
            <div className="bg-white p-3 rounded-full shadow-lg">
              <Brain className="w-12 h-12 text-blue-600" />
            </div>
          </div>
          <h1 className="text-4xl md:text-6xl font-bold text-gray-900 mb-4">
            Semantic <span className="text-blue-600">ArXiv</span>
          </h1>
          <p className="text-xl text-gray-600 max-w-2xl mx-auto mb-8">
            Search academic papers by meaning, not just keywords. Discover
            research through intelligent semantic understanding.
          </p>
        </div>

        {/* Search Section */}
        <div className="max-w-4xl mx-auto mb-12">
          <form
            onSubmit={handleSearch}
            className="bg-white rounded-lg shadow-xl p-8"
          >
            <div className="flex items-center space-x-4 mb-6">
              <Search className="w-6 h-6 text-gray-400" />
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="e.g., 'neural networks for computer vision' or 'machine learning optimization'"
                className="flex-1 text-lg border-0 outline-none text-gray-700 placeholder-gray-400"
                disabled={loading}
              />
              <button
                type="submit"
                disabled={loading}
                className="bg-blue-600 hover:bg-blue-700 disabled:bg-blue-400 text-white px-6 py-2 rounded-lg font-medium transition-colors"
              >
                {loading ? "Searching..." : "Search"}
              </button>
            </div>
            <div className="text-sm text-gray-500">
              Try: "attention mechanisms in computer vision" or "graph neural
              networks for drug discovery"
            </div>

            {error && (
              <div className="mt-4 p-3 bg-red-100 border border-red-400 text-red-700 rounded">
                {error}
              </div>
            )}
          </form>
        </div>

        {/* Search Results */}
        {hasSearched && (
          <div className="max-w-6xl mx-auto mb-12">
            {loading ? (
              <div className="text-center py-12">
                <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
                <p className="mt-4 text-gray-600">Searching papers...</p>
              </div>
            ) : results.length > 0 ? (
              <div>
                <div className="mb-6">
                  <h2 className="text-2xl font-bold text-gray-900">
                    Found {results.length} relevant papers
                  </h2>
                  <p className="text-gray-600">for "{query}"</p>
                </div>

                <div className="space-y-6">
                  {results.map((paper) => (
                    <PaperCard key={paper.id} paper={paper} />
                  ))}
                </div>
              </div>
            ) : (
              <div className="text-center py-12">
                <div className="bg-white rounded-lg shadow-lg p-8">
                  <Search className="w-16 h-16 text-gray-400 mx-auto mb-4" />
                  <h3 className="text-xl font-semibold text-gray-900 mb-2">
                    No papers found
                  </h3>
                  <p className="text-gray-600">
                    Try different keywords or broader search terms.
                  </p>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Features Section - Only show if no search has been performed */}
        {!hasSearched && (
          <div className="grid md:grid-cols-3 gap-8 mb-12">
            <div className="bg-white rounded-lg p-6 shadow-lg">
              <div className="bg-blue-100 w-12 h-12 rounded-lg flex items-center justify-center mb-4">
                <Sparkles className="w-6 h-6 text-blue-600" />
              </div>
              <h3 className="text-xl font-semibold text-gray-900 mb-2">
                Semantic Search
              </h3>
              <p className="text-gray-600">
                Find papers by meaning and concepts, not just exact keyword
                matches.
              </p>
            </div>

            <div className="bg-white rounded-lg p-6 shadow-lg">
              <div className="bg-green-100 w-12 h-12 rounded-lg flex items-center justify-center mb-4">
                <BookOpen className="w-6 h-6 text-green-600" />
              </div>
              <h3 className="text-xl font-semibold text-gray-900 mb-2">
                Latest Papers
              </h3>
              <p className="text-gray-600">
                Access research papers from arXiv across multiple domains.
              </p>
            </div>

            <div className="bg-white rounded-lg p-6 shadow-lg">
              <div className="bg-purple-100 w-12 h-12 rounded-lg flex items-center justify-center mb-4">
                <Brain className="w-6 h-6 text-purple-600" />
              </div>
              <h3 className="text-xl font-semibold text-gray-900 mb-2">
                AI-Powered
              </h3>
              <p className="text-gray-600">
                Powered by TF-IDF + SVD embeddings and FAISS for intelligent
                matching.
              </p>
            </div>
          </div>
        )}

        {/* Footer */}
        <div className="text-center text-gray-500">
          <p>Built using FastAPI, Next.js, and TF-IDF + SVD Embeddings</p>
        </div>
      </div>
    </div>
  );
}
