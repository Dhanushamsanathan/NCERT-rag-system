"""
NCERT RAG System v1 - Pipeline
==============================
Simple RAG implementation for NCERT textbooks (Classes 3-5)
"""

import os
import re
import pickle
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
        return text.strip()

    def _split_sentences(self, text: str) -> List[str]:
        """Simple sentence splitting"""
        # Split on sentence endings
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]


# ===================
# 3. Embedding Generator
# ===================

class EmbeddingGenerator:
    """Generate embeddings using Sentence Transformers"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        print(f"Loading embedding model: {model_name}")
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        print(f"Embedding dimension: {self.dimension}")

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Generate embeddings for list of texts"""
        print(f"Generating embeddings for {len(texts)} texts...")
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for a single query"""
        return self.model.encode([query], convert_to_numpy=True)[0]


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

    def generate(self, query: str, context: str) -> str:
        """Generate answer using retrieved context"""

        # Build prompt
        user_prompt = config.USER_PROMPT_TEMPLATE.format(
            context=context,
            question=query
        )

        # Call OpenRouter API
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://ncert-rag-system.local",
            "X-Title": "NCERT RAG System"
        }

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": config.SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 500
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
            return f"Error generating response: {e}"


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

        # Step 1: Embed query
        query_embedding = self.embedder.embed_query(question)

        # Step 2: Retrieve relevant chunks
        retrieved = self.vector_store.search(query_embedding, top_k)

        # Step 3: Build context
        context_parts = []
        for i, result in enumerate(retrieved, 1):
            source = f"[{result['metadata']['class']}/{result['metadata']['subject']}]"
            context_parts.append(f"{source} {result['content']}")

        context = "\n\n".join(context_parts)

        # Step 4: Generate answer
        answer = self.generator.generate(question, context)

        return {
            "question": question,
            "answer": answer,
            "sources": retrieved
        }


# ===================
# Quick Test
# ===================

if __name__ == "__main__":
    # Quick test
    rag = RAGPipeline()

    # Check if index exists
    if os.path.exists(config.VECTOR_DB_PATH):
        print("Loading existing index...")
        rag.load_index()
    else:
        print("Building new index...")
        rag.build_index()

    # Test query
    result = rag.query("What do plants need to grow?")
    print(f"\nQuestion: {result['question']}")
    print(f"\nAnswer: {result['answer']}")
    print(f"\nSources: {len(result['sources'])} chunks retrieved")
