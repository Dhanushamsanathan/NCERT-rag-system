"""
NCERT RAG System - Configuration
================================
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ===================
# API Configuration
# ===================
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct:free")

# ===================
# Data Configuration
# ===================
NCERT_DATA_PATH = os.getenv(
    "NCERT_DATA_PATH",
    "/home/dhanush/main_frames/ncert-textbook/final-ncert- dataset/output-single-image"
)

# Classes to include (for prototype: 3, 4, 5)
TARGET_CLASSES = ["class-3", "class-4", "class-5"]

# ===================
# Chunking Configuration
# ===================
CHUNK_SIZE = 500          # Characters per chunk
CHUNK_OVERLAP = 50        # Overlap between chunks

# ===================
# Embedding Configuration
# ===================
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Fast, good quality
EMBEDDING_DIMENSION = 384             # Output dimension

# ===================
# Retrieval Configuration
# ===================
TOP_K = 3                 # Number of chunks to retrieve

# ===================
# Vector DB Configuration
# ===================
VECTOR_DB_PATH = "vector_store"  # Directory to save FAISS index

# ===================
# LLM Prompt Template
# ===================
SYSTEM_PROMPT = """You are a helpful NCERT tutor for students in classes 3-5.

RULES:
1. Use ONLY the examples and information from the provided NCERT context
2. You can explain concepts based on the examples in the text
3. If the text shows examples without definitions, explain using those examples
4. Keep answers simple and appropriate for young students
5. Do not add information not in the context"""

USER_PROMPT_TEMPLATE = """CONTEXT from NCERT textbooks:
{context}

QUESTION: {question}

INSTRUCTIONS:
- Answer using the examples and information from the context
- If the text shows examples (like "bundle of sticks"), use them to explain
- You can infer simple explanations from the examples
- Keep answers simple for class 3-5 students

ANSWER:"""
