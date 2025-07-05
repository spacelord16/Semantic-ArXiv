# Semantic ArXiv

A Deep Learning-powered semantic search engine for academic papers from arXiv. Search by meaning and concepts, not just keywords.

## 🚀 Features

- **Semantic Search**: Find papers by meaning using state-of-the-art sentence transformers
- **Modern UI**: Clean, responsive interface built with Next.js and Tailwind CSS
- **Fast API**: High-performance backend with FastAPI and vector similarity search
- **Real-time Results**: Instant search results with similarity scoring
- **Multiple Categories**: Support for AI, Computer Vision, Machine Learning, and more

## 🛠️ Tech Stack

### Backend

- **FastAPI** - Modern Python web framework
- **Sentence Transformers** - State-of-the-art embeddings (all-MiniLM-L6-v2)
- **FAISS** - Vector similarity search
- **arXiv API** - Academic paper data source

### Frontend

- **Next.js 14** - React framework with App Router
- **TypeScript** - Type safety
- **Tailwind CSS** - Utility-first styling
- **Lucide Icons** - Beautiful icons

## 📁 Project Structure

```
semantic-arxiv/
├── backend/
│   ├── main.py              # FastAPI application
│   ├── config.py            # Configuration management
│   ├── requirements.txt     # Python dependencies
│   └── Dockerfile          # Backend Docker setup
├── frontend/
│   ├── app/
│   │   ├── page.tsx        # Home page
│   │   ├── layout.tsx      # Root layout
│   │   └── globals.css     # Global styles
│   ├── package.json        # Node.js dependencies
│   ├── next.config.js      # Next.js configuration
│   ├── tailwind.config.js  # Tailwind CSS config
│   └── Dockerfile          # Frontend Docker setup
├── docker-compose.yml      # Complete setup
└── README.md               # Project documentation
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+
- Docker (optional)

### Local Development

1. **Clone the repository**

   ```bash
   git clone <your-repo-url>
   cd semantic-arxiv
   ```

2. **Backend Setup**

   ```bash
   cd backend
   pip install -r requirements.txt
   uvicorn main:app --reload
   ```

3. **Frontend Setup**

   ```bash
   cd frontend
   npm install
   npm run dev
   ```

4. **Access the application**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000
   - API Docs: http://localhost:8000/docs

### Docker Setup

```bash
# Build and run with Docker Compose
docker-compose up --build

# Run in detached mode
docker-compose up -d
```

## 🔧 Configuration

Backend configuration is managed through environment variables:

```bash
# Vector Database
VECTOR_DB_TYPE=faiss
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Data Configuration
DATA_DIR=./data
ARXIV_CATEGORIES=cs.AI,cs.CV,cs.LG,cs.NE,cs.CL

# API Settings
DEBUG=True
HOST=0.0.0.0
PORT=8000
```

## 🎯 Roadmap

- [x] Project setup and architecture
- [ ] arXiv API integration
- [ ] Embedding generation pipeline
- [ ] Vector database implementation
- [ ] Search functionality
- [ ] Frontend search interface
- [ ] Performance optimization
- [ ] Deployment setup

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- arXiv for providing open access to research papers
- Hugging Face for the sentence transformers
- The open-source community for the amazing tools
