"""
Example configuration for the project. Copy this file to `config.py` and fill in the
real values, or better: set environment variables and keep `config.py` out of Git.

Do NOT commit your real `config.py` with keys to public repositories.
"""

# Secrets: set these as environment variables instead of hardcoding.
OPENAI_API_KEY = "your-openai-key-here"
PINECONE_API_KEY = "your-pinecone-key-here"

# Pinecone Index Configuration
PINECONE_INDEX_NAME = "rag-univera-chatbot-v2"
PINECONE_DIMENSION = 1024
PINECONE_METRIC = "cosine"
PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"

# Model Configuration
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large"
LLM_MODEL_NAME = "gpt-4o-mini"

# Search Configuration
DEFAULT_VECTOR_WEIGHT = 0.7
DEFAULT_BM25_WEIGHT = 0.3
DEFAULT_TOP_K = 50
DEFAULT_CONTEXT_CHARS = 3000
DEFAULT_CONTEXT_TOP_N = 5

# Document Paths (사용자 환경에 맞게 수정)
DOCUMENT_PATHS = {
    "products": r"C:\\path\\to\\RAG Database_Products Info",
    "company": r"C:\\path\\to\\RAG Database_Company Info"
}

# Chunking Configuration
CHUNK_SIZE = 500
CHUNK_OVERLAP = 75
