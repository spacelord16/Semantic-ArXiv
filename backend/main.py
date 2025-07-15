from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
from dotenv import load_dotenv
import logging
import pathlib

# Import our search functionality
from vector_database import search_papers

load_dotenv()

app = FastAPI(
    title="Semantic ArXiv API",
    description="A semantic search engine for academic papers",
    version="1.0.0",
)

# CORS middleware for frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://semantic-arxiv.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Pydantic models
class SearchQuery(BaseModel):
    query: str
    limit: Optional[int] = 10


class Paper(BaseModel):
    id: str
    title: str
    authors: List[str]
    abstract: str
    published: str
    url: str
    similarity_score: float


class SearchResponse(BaseModel):
    papers: List[Paper]
    query: str
    total_results: int


@app.get("/")
async def root():
    return {"message": "Semantic ArXiv API is running!"}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.post("/search", response_model=SearchResponse)
async def search_papers_endpoint(query: SearchQuery):
    """
    Search for papers using semantic similarity
    """
    try:
        # Debug information
        current_dir = os.getcwd()
        logger.info(f"Current working directory: {current_dir}")
        logger.info(f"Searching for: '{query.query}' with limit: {query.limit}")

        # Check if database files exist
        embeddings_dir = pathlib.Path("data/embeddings")
        index_file = embeddings_dir / "vector_index.faiss"
        model_file = embeddings_dir / "simple_embedding_model.pkl"

        logger.info(f"Embeddings directory exists: {embeddings_dir.exists()}")
        logger.info(f"Index file exists: {index_file.exists()}")
        logger.info(f"Model file exists: {model_file.exists()}")

        # Search using our vector database
        results = search_papers(
            query_text=query.query,
            k=query.limit,
            index_filename="vector_index",
            model_filename="simple_embedding_model",
        )

        logger.info(f"Raw search results count: {len(results)}")
        logger.info(f"First result (if any): {results[0] if results else 'None'}")

        # Transform results to Paper objects
        papers = []
        for result in results:
            metadata = result.get("metadata", {})

            # Handle the case where metadata might be missing fields
            paper = Paper(
                id=result.get("paper_id", ""),
                title=metadata.get("title", "Unknown Title"),
                authors=metadata.get("authors", []),
                abstract=metadata.get("abstract", "No abstract available"),
                published=metadata.get("published", ""),
                url=metadata.get("url", ""),
                similarity_score=result.get("similarity_score", 0.0),
            )
            papers.append(paper)

        logger.info(f"Found {len(papers)} results for query: '{query.query}'")

        return SearchResponse(
            papers=papers, query=query.query, total_results=len(papers)
        )

    except FileNotFoundError as e:
        logger.error(f"Database files not found: {e}")
        raise HTTPException(
            status_code=503,
            detail="Search service temporarily unavailable. Database files not found.",
        )
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(
            status_code=500, detail=f"Internal server error during search: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
