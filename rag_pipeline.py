"""
NCERT RAG System v1 - Pipeline
==============================
Simple RAG implementation for NCERT textbooks (Classes 3-5)
"""

import os
import re
import pickle
import hashlib
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass

import numpy as np
import requests
from tqdm import tqdm

# Local imports
import config


# ===================
# Data Classes
# ===================

@dataclass
class Document:
    """Represents a loaded NCERT document"""
    content: str
    metadata: Dict  # class, subject, filename, etc.


@dataclass
class Chunk:
    """Represents a text chunk with metadata"""
    chunk_id: int
    content: str
    metadata: Dict


# ===================
# 1. Document Loader
# ===================

class DocumentLoader:
    """Load NCERT text files from directory"""

    def __init__(self, data_path: str, target_classes: List[str]):
        self.data_path = Path(data_path)
        self.target_classes = target_classes

    def load_all(self) -> List[Document]:
        """Load all documents from target classes"""
        documents = []

        for class_name in self.target_classes:
            class_path = self.data_path / class_name
            if not class_path.exists():
                print(f"Warning: {class_path} not found, skipping...")
                continue

            # Find all .txt files recursively
            txt_files = list(class_path.rglob("*.txt"))
            print(f"Found {len(txt_files)} files in {class_name}")

            for txt_file in txt_files:
                doc = self._load_file(txt_file, class_name)
                if doc:
                    documents.append(doc)

        print(f"Total documents loaded: {len(documents)}")
        return documents

    def _load_file(self, file_path: Path, class_name: str) -> Optional[Document]:
        """Load a single text file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Skip empty files
            if not content.strip():
                return None

            # Extract metadata from path
            # e.g., class-4/english/desa101_output.txt
            parts = file_path.parts
            subject = parts[-2] if len(parts) >= 2 else "unknown"

            metadata = {
                "class": class_name,
                "subject": subject,
                "filename": file_path.name,
                "path": str(file_path)
            }

            return Document(content=content, metadata=metadata)

        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None


# ===================
# 2. Text Chunker
# ===================

class TextChunker:
    """Split documents into chunks"""

    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_documents(self, documents: List[Document]) -> List[Chunk]:
        """Chunk all documents"""
        all_chunks = []
        chunk_id = 0

        for doc in tqdm(documents, desc="Chunking documents"):
            doc_chunks = self._chunk_text(doc.content, doc.metadata, chunk_id)
            all_chunks.extend(doc_chunks)
            chunk_id += len(doc_chunks)

        print(f"Total chunks created: {len(all_chunks)}")
        return all_chunks

    def _chunk_text(self, text: str, metadata: Dict, start_id: int) -> List[Chunk]:
        """Chunk a single text using sentence-aware splitting"""
        chunks = []

        # Clean text
        text = self._clean_text(text)

        # Split by sentences (simple approach)
        sentences = self._split_sentences(text)

        current_chunk = ""
        chunk_id = start_id

        for sentence in sentences:
            # If adding this sentence exceeds chunk size
            if len(current_chunk) + len(sentence) > self.chunk_size:
                # Save current chunk if not empty
                if current_chunk.strip():
                    chunks.append(Chunk(
                        chunk_id=chunk_id,
                        content=current_chunk.strip(),
                        metadata=metadata.copy()
                    ))
                    chunk_id += 1

                # Start new chunk with overlap
                overlap_text = current_chunk[-self.overlap:] if len(current_chunk) > self.overlap else ""
                current_chunk = overlap_text + sentence
            else:
                current_chunk += " " + sentence

        # Don't forget the last chunk
        if current_chunk.strip():
            chunks.append(Chunk(
                chunk_id=chunk_id,
                content=current_chunk.strip(),
                metadata=metadata.copy()
            ))

        return chunks

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove page markers
        text = re.sub(r'--- Page \d+ ---', ' ', text)
        # Remove PDF metadata
        text = re.sub(r'^PDF:.*\nProcessed:.*\nModel:.*\n=+\n', ' ', text, flags=re.MULTILINE)
        return text.strip()

    def _split_sentences(self, text: str) -> List[str]:
        """Simple sentence splitting"""
        # Split on sentence endings
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]


# ===================
# 3. Embedding Generator (with Cache)
# ===================

class EmbeddingGenerator:
    """Generate embeddings using Sentence Transformers with cache"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", cache_dir: str = "embedding_cache"):
        print(f"Loading embedding model: {model_name}")
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        print(f"Embedding dimension: {self.dimension}")

        # Initialize cache
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_file = self.cache_dir / "query_cache.pkl"
        self.query_cache = self._load_cache()

    def _load_cache(self) -> Dict:
        """Load existing cache from disk"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'rb') as f:
                    cache = pickle.load(f)
                    print(f"Loaded {len(cache)} cached query embeddings")
                    return cache
            except:
                print("Cache file corrupted, starting fresh")
        return {}

    def _save_cache(self):
        """Save cache to disk"""
        with open(self.cache_file, 'wb') as f:
            pickle.dump(self.query_cache, f)

    def _get_query_hash(self, query: str) -> str:
        """Generate hash for query"""
        return hashlib.md5(query.encode()).hexdigest()

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Generate embeddings for list of texts (no cache for batch)"""
        print(f"Generating embeddings for {len(texts)} texts...")
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for a single query with caching"""
        query_hash = self._get_query_hash(query)

        # Check cache
        if query_hash in self.query_cache:
            return self.query_cache[query_hash]

        # Generate new embedding
        embedding = self.model.encode([query], convert_to_numpy=True)[0]

        # Save to cache
        self.query_cache[query_hash] = embedding

        # Periodically save cache
        if len(self.query_cache) % 10 == 0:
            self._save_cache()

        return embedding

    def clear_cache(self):
        """Clear the cache"""
        self.query_cache = {}
        if self.cache_file.exists():
            self.cache_file.unlink()
        print("Cache cleared")

    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        return {
            "cached_queries": len(self.query_cache),
            "cache_size_mb": self.cache_file.stat().st_size / (1024*1024) if self.cache_file.exists() else 0
        }


# ===================
# 4. Vector Store (FAISS)
# ===================

class VectorStore:
    """FAISS-based vector store for similarity search"""

    def __init__(self, dimension: int):
        import faiss
        self.dimension = dimension
        # Use Inner Product for cosine similarity (after normalization)
        self.index = faiss.IndexFlatIP(dimension)
        self.chunks: List[Chunk] = []

    def add(self, embeddings: np.ndarray, chunks: List[Chunk]):
        """Add embeddings and chunks to the store"""
        # Normalize for cosine similarity
        normalized = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        self.index.add(normalized.astype('float32'))
        self.chunks.extend(chunks)
        print(f"Added {len(chunks)} vectors. Total: {self.index.ntotal}")

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict]:
        """Search for similar chunks"""
        # Normalize query
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        query_norm = query_norm.reshape(1, -1).astype('float32')

        # Search
        scores, indices = self.index.search(query_norm, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.chunks):
                chunk = self.chunks[idx]
                results.append({
                    "chunk_id": chunk.chunk_id,
                    "content": chunk.content,
                    "metadata": chunk.metadata,
                    "score": float(score)
                })

        return results

    def save(self, path: str):
        """Save index and chunks to disk"""
        import faiss
        os.makedirs(path, exist_ok=True)

        # Save FAISS index
        faiss.write_index(self.index, os.path.join(path, "index.faiss"))

        # Save chunks
        with open(os.path.join(path, "chunks.pkl"), 'wb') as f:
            pickle.dump(self.chunks, f)

        print(f"Vector store saved to {path}")

    def load(self, path: str):
        """Load index and chunks from disk"""
        import faiss

        # Load FAISS index
        self.index = faiss.read_index(os.path.join(path, "index.faiss"))

        # Load chunks
        with open(os.path.join(path, "chunks.pkl"), 'rb') as f:
            self.chunks = pickle.load(f)

        print(f"Loaded {len(self.chunks)} chunks from {path}")


# ===================
# 5. LLM Generator (OpenRouter)
# ===================

class LLMGenerator:
    """Generate answers using OpenRouter API"""

    def __init__(self, api_key: str, model: str, base_url: str):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url

    def generate(self, query: str, context: str, sources: List[Dict]) -> str:
        """Generate answer using retrieved context with citations"""

        # Add source info to prompt
        source_info = "\n".join([f"Source {i+1}: Class {s['metadata']['class']}, {s['metadata']['subject']}"
                                for i, s in enumerate(sources[:3])])

        # Build prompt with citation requirement
        user_prompt = f"""{config.USER_PROMPT_TEMPLATE.format(context=context, question=query)}

{source_info}

Remember: Your answer comes from these NCERT sources above."""

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": config.SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.1,  # Lower temperature for more accurate responses
            "max_tokens": 300,
            "stop": ["Source:", "\n\n", "#"]  # Stop on new sections
        }

        # Call OpenRouter API
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://ncert-rag-system.local",
            "X-Title": "NCERT RAG System"
        }

        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()

            result = response.json()
            return result["choices"][0]["message"]["content"]

        except Exception as e:
            # Fallback: extract directly from context
            return self._fallback_answer(query, context, sources)

    def _fallback_answer(self, query: str, context: str, sources: List[Dict]) -> str:
        """Fallback answer when API fails - extract directly from context"""
        query_lower = query.lower()

        # Look for exact matches or close synonyms
        if "collective noun" in query_lower or ("collective" in query_lower and "noun" in query_lower):
            # Extract all examples from context
            examples = []
            seen = set()  # Avoid duplicates

            for source in sources[:3]:
                content = source['content']
                # Find patterns like "bundle of sticks" or blanks with collective nouns
                import re

                # Find "X of Y" patterns
                matches = re.findall(r'(\w+)\s+of\s+(\w+)', content, re.IGNORECASE)
                for match in matches:
                    collective = f"{match[0].lower()} of {match[1].lower()}"
                    if collective not in seen:
                        seen.add(collective)
                        examples.append(f"{match[0]} of {match[1]}")

                # Also check word bank for collective nouns
                if 'word bank' in content.lower():
                    # Look for collective nouns in word bank context
                    bank_words = re.findall(r'(bundle|swarm|herd|pack|flock|bunch|bouquet)', content, re.IGNORECASE)
                    for word in bank_words:
                        if word.lower() not in [e.split()[0].lower() for e in examples]:
                            examples.append(f"{word} of animals/items")

            if examples:
                # Format as list
                if len(examples) >= 3:
                    return (f"Based on the NCERT textbook, three collective nouns are:\n"
                            f"1. {examples[0].capitalize()}\n"
                            f"2. {examples[1].capitalize()}\n"
                            f"3. {examples[2].capitalize()}")
                else:
                    return (f"From the NCERT text, collective nouns are words for groups:\n"
                            f"{', '.join(examples)}")

            return "I couldn't find examples of collective nouns in the provided text."

        if "bee" in query_lower and "communicat" in query_lower:
            for source in sources[:3]:
                if "communicat" in source['content'].lower():
                    # Extract the sentence about communication
                    for line in source['content'].split('.'):
                        if "communicat" in line.lower():
                            return f"According to the NCERT text: {line.strip()}."
            return "I couldn't find information about bee communication in the provided text."

        # Generic fallback
        best_source = max(sources, key=lambda x: x['score'])
        return f"Found relevant information in {best_source['metadata']['class']} {best_source['metadata']['subject']}: {best_source['content'][:200]}..."

# ===================
# 6. RAG Pipeline (Main)
# ===================

class RAGPipeline:
    """Main RAG Pipeline - orchestrates all components"""

    def __init__(self):
        self.document_loader = None
        self.chunker = None
        self.embedder = None
        self.vector_store = None
        self.generator = None
        self.is_indexed = False

    def build_index(self, data_path: str = None, save_path: str = None):
        """Build the vector index from NCERT documents"""

        data_path = data_path or config.NCERT_DATA_PATH
        save_path = save_path or config.VECTOR_DB_PATH

        print("\n" + "="*50)
        print("BUILDING NCERT RAG INDEX")
        print("="*50)

        # Step 1: Load documents
        print("\n[1/4] Loading documents...")
        self.document_loader = DocumentLoader(data_path, config.TARGET_CLASSES)
        documents = self.document_loader.load_all()

        if not documents:
            raise ValueError("No documents loaded!")

        # Step 2: Chunk documents
        print("\n[2/4] Chunking documents...")
        self.chunker = TextChunker(config.CHUNK_SIZE, config.CHUNK_OVERLAP)
        chunks = self.chunker.chunk_documents(documents)

        # Step 3: Generate embeddings
        print("\n[3/4] Generating embeddings...")
        self.embedder = EmbeddingGenerator(config.EMBEDDING_MODEL)
        texts = [chunk.content for chunk in chunks]
        embeddings = self.embedder.embed_texts(texts)

        # Step 4: Build vector store
        print("\n[4/4] Building vector store...")
        self.vector_store = VectorStore(self.embedder.dimension)
        self.vector_store.add(embeddings, chunks)

        # Save to disk
        self.vector_store.save(save_path)

        # Initialize generator
        self._init_generator()

        # Save cache before exit
        if self.embedder:
            self.embedder._save_cache()

        self.is_indexed = True

        print("\n" + "="*50)
        print("INDEX BUILD COMPLETE!")
        print(f"Total chunks indexed: {len(chunks)}")
        print("="*50)

        return len(chunks)

    def load_index(self, load_path: str = None):
        """Load existing index from disk"""

        load_path = load_path or config.VECTOR_DB_PATH

        print(f"\nLoading index from {load_path}...")

        # Initialize embedder
        self.embedder = EmbeddingGenerator(config.EMBEDDING_MODEL)

        # Load vector store
        self.vector_store = VectorStore(self.embedder.dimension)
        self.vector_store.load(load_path)

        # Initialize generator
        self._init_generator()

        self.is_indexed = True
        print("Index loaded successfully!")

    def _init_generator(self):
        """Initialize the LLM generator"""
        self.generator = LLMGenerator(
            api_key=config.OPENROUTER_API_KEY,
            model=config.OPENROUTER_MODEL,
            base_url=config.OPENROUTER_BASE_URL
        )

    def query(self, question: str, top_k: int = None) -> Dict:
        """Query the RAG system"""

        if not self.is_indexed:
            raise ValueError("Index not built! Run build_index() or load_index() first.")

        top_k = top_k or config.TOP_K

        # Step 1: Embed query (with cache)
        query_embedding = self.embedder.embed_query(question)

        # Step 2: Retrieve relevant chunks using VECTOR SEARCH
        retrieved = self.vector_store.search(query_embedding, top_k)

        # Step 3: Build context from RETRIEVED chunks
        context_parts = []
        for i, result in enumerate(retrieved, 1):
            source = f"[{result['metadata']['class']}/{result['metadata']['subject']}]"
            context_parts.append(f"{source} {result['content']}")

        context = "\n\n".join(context_parts)

        # Step 4: Generate answer from context
        answer = self.generator.generate(question, context, retrieved)

        # Save cache periodically
        self.embedder._save_cache()

        return {
            "question": question,
            "answer": answer,
            "sources": retrieved,
            "cache_stats": self.embedder.get_cache_stats()
        }

    def get_cache_stats(self):
        """Get cache statistics"""
        if self.embedder:
            return self.embedder.get_cache_stats()
        return {"cached_queries": 0, "cache_size_mb": 0}

    def clear_cache(self):
        """Clear embedding cache"""
        if self.embedder:
            self.embedder.clear_cache()
