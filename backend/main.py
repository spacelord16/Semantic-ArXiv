from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="Semantic ArXiv API",
    description="A semantic search engine for academic papers",
    version="1.0.0"
)

# CORS middleware for frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://semantic-arxiv.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
async def search_papers(query: SearchQuery):
    """
    Search for papers using semantic similarity
    """
    # TODO: Implement semantic search logic
    return SearchResponse(
        papers=[],
        query=query.query,
        total_results=0
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 