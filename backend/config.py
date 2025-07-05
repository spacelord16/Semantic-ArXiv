import os
from typing import List
from dotenv import load_dotenv

load_dotenv()

class Config:
    # ArXiv API Configuration
    ARXIV_API_BASE_URL = os.getenv("ARXIV_API_BASE_URL", "http://export.arxiv.org/api")
    
    # Vector Database Configuration
    VECTOR_DB_TYPE = os.getenv("VECTOR_DB_TYPE", "faiss")  # Options: faiss, pinecone
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "semantic-arxiv")
    
    # Model Configuration
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "384"))
    
    # Data Configuration
    DATA_DIR = os.getenv("DATA_DIR", "./data")
    MAX_PAPERS_PER_CATEGORY = int(os.getenv("MAX_PAPERS_PER_CATEGORY", "10000"))
    ARXIV_CATEGORIES = os.getenv("ARXIV_CATEGORIES", "cs.AI,cs.CV,cs.LG,cs.NE,cs.CL").split(",")
    
    # API Configuration
    DEBUG = os.getenv("DEBUG", "True").lower() == "true"
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "8000"))

config = Config() 