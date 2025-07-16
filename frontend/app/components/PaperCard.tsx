import { ExternalLink, User, Calendar, Star, BookOpen } from "lucide-react";

interface Paper {
  id: string;
  title: string;
  authors: string[];
  abstract: string;
  published: string;
  url: string;
  similarity_score: number;
}

interface PaperCardProps {
  paper: Paper;
}

export default function PaperCard({ paper }: PaperCardProps) {
  const formatDate = (dateString: string) => {
    try {
      return new Date(dateString).toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric'
      });
    } catch {
      return dateString;
    }
  };

  const truncateAbstract = (abstract: string, maxLength: number = 300) => {
    if (abstract.length <= maxLength) return abstract;
    return abstract.substr(0, maxLength) + '...';
  };

  const getScoreColor = (score: number) => {
    if (score >= 0.7) return 'text-green-600 bg-green-50';
    if (score >= 0.5) return 'text-yellow-600 bg-yellow-50';
    return 'text-gray-600 bg-gray-50';
  };

  const getScoreLabel = (score: number) => {
    if (score >= 0.7) return 'Highly Relevant';
    if (score >= 0.5) return 'Relevant';
    return 'Somewhat Relevant';
  };

  return (
    <div className="bg-white rounded-lg shadow-lg p-6 hover:shadow-xl transition-shadow border border-gray-100">
      <div className="flex items-start justify-between mb-4">
        <div className="flex-1">
          <h3 className="text-xl font-semibold text-gray-900 mb-3 leading-tight">
            {paper.title}
          </h3>
          
          <div className="flex flex-wrap items-center gap-4 text-sm text-gray-500 mb-4">
            <div className="flex items-center space-x-1">
              <User className="w-4 h-4" />
              <span className="font-medium">
                {paper.authors.slice(0, 3).join(', ')}
                {paper.authors.length > 3 && ` +${paper.authors.length - 3} more`}
              </span>
            </div>
            
            <div className="flex items-center space-x-1">
              <Calendar className="w-4 h-4" />
              <span>{formatDate(paper.published)}</span>
            </div>
            
            <div className={`flex items-center space-x-1 px-2 py-1 rounded-full text-xs font-medium ${getScoreColor(paper.similarity_score)}`}>
              <Star className="w-3 h-3" />
              <span>{(paper.similarity_score * 100).toFixed(1)}% • {getScoreLabel(paper.similarity_score)}</span>
            </div>
          </div>
        </div>
        
        <div className="flex items-center space-x-2 ml-4">
          <a
            href={paper.url}
            target="_blank"
            rel="noopener noreferrer"
            className="p-2 text-blue-600 hover:text-blue-800 hover:bg-blue-50 rounded-lg transition-colors"
            title="View on arXiv"
          >
            <ExternalLink className="w-5 h-5" />
          </a>
        </div>
      </div>
      
      <div className="prose prose-sm max-w-none">
        <p className="text-gray-700 leading-relaxed">
          {truncateAbstract(paper.abstract)}
        </p>
      </div>
      
      <div className="mt-4 pt-4 border-t border-gray-100">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-1 text-xs text-gray-500">
            <BookOpen className="w-3 h-3" />
            <span>Paper ID: {paper.id}</span>
          </div>
          
          <a
            href={paper.url}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center space-x-1 text-sm text-blue-600 hover:text-blue-800 font-medium"
          >
            <span>Read Paper</span>
            <ExternalLink className="w-4 h-4" />
          </a>
        </div>
      </div>
    </div>
  );
}