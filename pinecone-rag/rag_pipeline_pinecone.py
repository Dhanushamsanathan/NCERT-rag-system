#!/usr/bin/env python3
"""
Fast RAG Pipeline with Pinecone
================================
Uses Pinecone for lightning-fast vector search
Expected response time: <2 seconds
"""

import os
import pickle
import numpy as np
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import requests
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class PineconeRAGPipeline:
    def __init__(self):
        self.embed_model = None
        self.index = None
        self.client = None
        self.docs_cache = {}  # Cache for documents

    def load_index(self, index_path: str = "vector_store"):
        """Load Pinecone index"""
        print("Connecting to Pinecone...")

        # Get API key from environment or .env
        api_key = os.getenv('PINECONE_API_KEY')
        if not api_key:
            # Try to load from .env file
            env_file = os.path.join(os.path.dirname(__file__), '.env')
            if os.path.exists(env_file):
                with open(env_file, 'r') as f:
                    for line in f:
                        if line.startswith('PINECONE_API_KEY='):
                            api_key = line.split('=', 1)[1].strip()
                            break

        if not api_key:
            raise ValueError("PINECONE_API_KEY not found. Please set it in environment or .env file")

        # Initialize Pinecone
        pc = Pinecone(api_key=api_key)
        index_name = "ncert-index"

        # Check if index exists
        if index_name not in pc.list_indexes().names():
            print(f"Index '{index_name}' not found. Please run setup_pinecone.py first!")
            return False

        # Connect to index
        self.index = pc.Index(index_name)
        print(f"✅ Connected to Pinecone index: {index_name}")

        # Load embedding model
        print("Loading embedding model...")
        self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.dimension = self.embed_model.get_sentence_embedding_dimension()

        # OpenRouter client not needed - using requests directly

        # Load documents for context
        docs_file = os.path.join(index_path, 'documents.pkl')
        if os.path.exists(docs_file):
            with open(docs_file, 'rb') as f:
                data = pickle.load(f)
                self.docs_cache = data['chunks']
            print(f"✅ Loaded {len(self.docs_cache)} documents to cache")

        return True

    def _get_embedding(self, text: str) -> np.ndarray:
        """Get text embedding"""
        return self.embed_model.encode([text])[0]

    def _extract_metadata(self, metadata: Dict) -> Dict[str, Any]:
        """Extract metadata from Pinecone response"""
        return {
            'class': metadata.get('class', 'Unknown'),
            'subject': metadata.get('subject', 'General'),
            'content': metadata.get('text', '')
        }

    def query(self, question: str, top_k: int = 3) -> Dict[str, Any]:
        """Query using Pinecone for fast search"""
        start_time = time.time()

        # Get question embedding
        query_embedding = self._get_embedding(question)

        # Search Pinecone (very fast - ~50ms)
        results = self.index.query(
            vector=query_embedding.tolist(),
            top_k=top_k,
            include_metadata=True,
            include_values=False
        )

        # Process results
        sources = []
        for match in results['matches']:
            metadata = self._extract_metadata(match['metadata'])
            sources.append({
                'content': metadata['content'],
                'metadata': {
                    'class': metadata['class'],
                    'subject': metadata['subject']
                },
                'score': match['score']
            })

        # Generate response if we have good matches
        answer = None
        context = None
        max_score = sources[0]['score'] if sources else 0

        if sources and max_score > 0.5:  # Pinecone uses cosine similarity
            # Build context from top 2 results (for speed)
            context_parts = []
            for i, source in enumerate(sources[:2], 1):
                content = source['content']
                if len(content) > 500:
                    content = content[:500] + "..."
                context_parts.append(f"Source {i}: {content}")

            context = "\n\n".join(context_parts)
            answer = self._generate_response(question, context)

        response_time = time.time() - start_time
        print(f"Query completed in {response_time:.2f} seconds")

        return {
            'answer': answer,
            'sources': sources if answer else [],
            'max_score': max_score,
            'context': context
        }

    def _generate_response(self, query: str, context: str) -> str:
        """Generate response using OpenRouter API"""
        system_prompt = """You are an NCERT tutor for Classes 1–7.
Use simple, child-friendly language.
Answer only from the provided context.
If the answer is not in the context, say: "This information is not available in the given material."
Reply Rules:
Language Style
Use easy words, short sentences.
Explain hard terms simply.
Keep a warm, positive, encouraging tone.
Response Structure
Use bullet points.
Give small examples or simple steps when helpful.
Highlight key ideas clearly.
Behavior
Be patient, gentle, and supportive.
Never criticize.
Re-explain in simpler words if the child seems confused.
Encourage curiosity ("Good question!").
Content Limits
Use only context; add nothing extra.
If missing info → say: "This information is not available in the given material."
Safety
Avoid sensitive/advanced topics.
Use only school-safe, child-friendly examples."""

        # Get API key from environment
        api_key = os.getenv('OPENROUTER_API_KEY')
        model = os.getenv('OPENROUTER_MODEL', 'microsoft/wizardlm-2-8x22b')

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"}
            ],
            "max_tokens": 80,
            "temperature": 0.1
        }

        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=data
            )
            response.raise_for_status()
            result = response.json()
            return result['choices'][0]['message']['content'].strip()
        except Exception as e:
            print(f"LLM API Error: {e}")
            return f"Error generating response: {str(e)}"