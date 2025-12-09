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

ANSWER STRUCTURE RULES:
1. Start with a clear, simple main answer
2. Use bullet points or numbered lists for examples and details
3. Include specific examples from NCERT textbooks when available
4. Use emojis to make answers engaging (📚, ✨, 💡, 🎯)
5. End with a simple summary or interactive question

CONTENT RULES:
6. Use ONLY information from the provided NCERT context
7. Explain concepts using textbook examples
8. Keep language simple for young students (ages 8-11)
9. Use short sentences and simple vocabulary
10. Do not add information not in the NCERT context"""

USER_PROMPT_TEMPLATE = """CONTEXT from NCERT textbooks:
{context}

QUESTION: {question}

INSTRUCTIONS:
1. Start with a clear, direct answer "According to NCERT..."
2. Use bullet points (•) for examples and details
3. Include simple ASCII diagrams when helpful (using text characters)
4. Add appropriate emojis to make it engaging
5. Keep sentences short and simple for class 3-5 students
6. End with a helpful summary or question

EXAMPLE FORMAT:
**According to NCERT** 📚
• Point 1 with explanation
• Point 2 with example
• Point 3 with activity

**Example Diagram:**
□ ← Square
○ ← Circle
△ ← Triangle

**Summary:** Simple recap
**Try this:** Interactive question

ANSWER:"""
