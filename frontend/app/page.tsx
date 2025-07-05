import { Search, BookOpen, Brain, Sparkles } from "lucide-react";

export default function Home() {
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
          <div className="bg-white rounded-lg shadow-xl p-8">
            <div className="flex items-center space-x-4 mb-6">
              <Search className="w-6 h-6 text-gray-400" />
              <input
                type="text"
                placeholder="e.g., 'how to make neural networks faster without losing accuracy'"
                className="flex-1 text-lg border-0 outline-none text-gray-700 placeholder-gray-400"
              />
              <button className="bg-blue-600 hover:bg-blue-700 text-white px-6 py-2 rounded-lg font-medium transition-colors">
                Search
              </button>
            </div>
            <div className="text-sm text-gray-500">
              Try: "attention mechanisms in computer vision" or "graph neural
              networks for drug discovery"
            </div>
          </div>
        </div>

        {/* Features Section */}
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
              Access the most recent research from arXiv across multiple
              domains.
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
              Powered by state-of-the-art transformer models for intelligent
              matching.
            </p>
          </div>
        </div>

        {/* Footer */}
        <div className="text-center text-gray-500">
          <p>Built with ❤️ using FastAPI, Next.js, and Sentence Transformers</p>
        </div>
      </div>
    </div>
  );
}
