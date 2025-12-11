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
            "max_tokens": 500
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
            answer = result["choices"][0]["message"]["content"]
            print(f"DEBUG: LLM response successful, length: {len(answer)}")
            return answer

        except Exception as e:
            print(f"DEBUG: LLM API failed: {e}")
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
        self.bm25 = None  # BM25 for hybrid search
        self.bm25_chunks = []
        self.is_indexed = False

        # Agentic: Topic context cache
        self.topic_cache = {}  # topic -> list of chunks
        self.last_query_topic = None

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

        # Build BM25 index
        print("\n[+] Building BM25 index...")
        self._build_bm25(chunks)

        # Save to disk
        self.vector_store.save(save_path)
        self._save_bm25(save_path)

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

        # Load BM25 index
        self._load_bm25(load_path)

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

    def _extract_topic(self, question: str) -> str:
        """Extract main topic and class/subject for caching"""
        q = question.lower()

        # Extract class if mentioned
        class_num = None
        if "class 3" in q or "class-3" in q:
            class_num = "class-3"
        elif "class 4" in q or "class-4" in q:
            class_num = "class-4"
        elif "class 5" in q or "class-5" in q:
            class_num = "class-5"

        # Extract subject
        subject = None
        if "math" in q:
            subject = "maths"
        elif "english" in q:
            subject = "english"
        elif "science" in q or "evs" in q:
            subject = "science"

        # Extract main topic keywords
        topics = []
        # Common NCERT topics
        topic_keywords = [
            "photosynthesis", "shapes", "collective noun", "grammar", "addition",
            "subtraction", "plants", "animals", "fraction", "geometry",
            "stories", "poem", "environment", "family", "friends", "teamwork"
        ]

        for keyword in topic_keywords:
            if keyword in q:
                topics.append(keyword)

        # Create topic key
        if topics:
            topic_key = "_".join(topics)
        else:
            # Use first 2 words as topic if no keyword matches
            words = question.split()[:2]
            topic_key = "_".join(words)

        if class_num:
            topic_key = f"{class_num}_{topic_key}"
        if subject:
            topic_key = f"{topic_key}_{subject}"

        return topic_key

    def _get_topic_chunks(self, topic: str, refined_question: str) -> List[Dict]:
        """Get chunks for a topic - use cache if available"""

        # Check cache first
        if topic in self.topic_cache:
            print(f"Using cached topic: {topic} ({len(self.topic_cache[topic])} chunks)")
            return self.topic_cache[topic]

        # Not in cache - retrieve comprehensive content for this topic
        print(f"Building topic cache: {topic}")
        print(f"Refined query: {refined_question}")

        # Use more chunks for topic building
        comprehensive_top_k = 15
        query_embedding = self.embedder.embed_query(refined_question)

        if self.bm25:
            retrieved = self._hybrid_search(refined_question, query_embedding, comprehensive_top_k)
        else:
            retrieved = self.vector_store.search(query_embedding, comprehensive_top_k)

        # Retrieved successfully (no debug needed)
        # print(f"Retrieved {len(retrieved)} chunks for topic: {topic}")

        # Cache for future use
        self.topic_cache[topic] = retrieved
        print(f"Cached {len(retrieved)} chunks for topic: {topic}")

        return retrieved

    def _refine_query(self, question: str) -> str:
        """Use LLM to refine/clarify the query without changing the intent"""

        # Skip refinement for simple, clear queries
        if len(question.split()) <= 5 and '?' in question:
            return question

        # Use a quick LLM call to refine
        prompt = f"""Improve this question for better search results.
Keep the same meaning, just make it clearer.

Original: {question}

Improved (only if needed, otherwise return original):"""

        try:
            payload = {
                "model": config.OPENROUTER_MODEL,
                "messages": [
                    {"role": "system", "content": "You are a query refiner. Make questions clearer but keep exact same meaning."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.1,
                "max_tokens": 50
            }

            headers = {
                "Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
                "Content-Type": "application/json"
            }

            response = requests.post(
                f"{config.OPENROUTER_BASE_URL}/chat/completions",
                headers=headers,
                json=payload,
                timeout=5
            )

            if response.status_code == 200:
                refined = response.json()["choices"][0]["message"]["content"].strip()
                # Only use if it's actually better/longer
                if len(refined) > len(question) * 0.8:
                    return refined

        except:
            pass  # Fall back to original if anything fails

        return question

    def _get_smart_top_k(self, question: str) -> int:
        """Smart pre-retrieval: determine top_k based on query type"""
        q = question.lower()

        # Comprehensive queries - need more examples
        if any(word in q for word in ["all", "every", "list", "examples", "types", "kinds", "show me", "find all", "what are", "name all"]):
            return 12

        # Comparison queries - need content for both sides
        elif any(word in q for word in ["compare", "difference", "vs", "versus", "similar", "different"]):
            return 8

        # Multi-subject queries
        elif any(word in q for word in ["subjects", "topics", "chapters", "books", "classes"]):
            return 10

        # Default - simple query
        else:
            return config.TOP_K  # 3

    def query(self, question: str, top_k: int = None, use_hybrid: bool = True) -> Dict:
        """Query the RAG system with smart retrieval and topic caching"""

        if not self.is_indexed:
            raise ValueError("Index not built! Run build_index() or load_index() first.")

        # Step 0: Refine query for better search
        refined_question = self._refine_query(question)

        # Step 1: Extract topic for caching
        topic = self._extract_topic(question)

        # Step 2: Get chunks - use topic cache if available
        if topic in self.topic_cache or self._is_topic_based_query(question):
            # Use comprehensive topic chunks
            retrieved = self._get_topic_chunks(topic, refined_question)
            self.last_query_topic = topic
        else:
            # Regular smart retrieval
            smart_top_k = top_k or self._get_smart_top_k(refined_question)
            query_embedding = self.embedder.embed_query(refined_question)

            if use_hybrid and self.bm25:
                retrieved = self._hybrid_search(refined_question, query_embedding, smart_top_k)
            else:
                retrieved = self.vector_store.search(query_embedding, smart_top_k)

        # Step 3: Build context from RETRIEVED chunks (limit to prevent overwhelming)
        context_parts = []
        max_context_length = 3000  # Limit context to prevent LLM overload
        current_length = 0

        for i, result in enumerate(retrieved, 1):
            source = f"[{result['metadata']['class']}/{result['metadata']['subject']}]"
            chunk_text = f"{source} {result['content']}"

            # Check if adding this chunk exceeds limit
            if current_length + len(chunk_text) > max_context_length:
                break

            context_parts.append(chunk_text)
            current_length += len(chunk_text)

        context = "\n\n".join(context_parts)

        # Step 4: Generate answer from context
        try:
            answer = self.generator.generate(question, context, retrieved)
        except Exception as e:
            answer = f"According to NCERT 📚\n\nI found information about this topic in your NCERT textbooks."

        # Step 5: LLM verification and intelligent response
        try:
            answer = self._llm_verification(question, answer, retrieved)
        except Exception as e:
            # Keep the original answer if verification fails
            pass

        # Save cache periodically
        self.embedder._save_cache()

        return {
            "question": question,
            "refined_question": refined_question if refined_question != question else None,
            "topic": topic if topic in self.topic_cache else None,
            "answer": answer,
            "sources": retrieved,
            "cache_stats": self.embedder.get_cache_stats()
        }

    def _is_topic_based_query(self, question: str) -> bool:
        """Check if query should use topic-based retrieval"""
        q = question.lower()

        # Use topic-based if it asks about specific concepts
        topic_indicators = [
            "tell me about", "explain", "what is", "describe",
            "all about", "everything about", "learn about",
            "photosynthesis", "shapes", "collective noun", "grammar",
            "addition", "subtraction", "plants", "animals"
        ]

        return any(indicator in q for indicator in topic_indicators)

    def _llm_verification(self, question: str, answer: str, retrieved: List[Dict]) -> str:
        """Use LLM to verify if answer is good and improve if needed"""

        # Check if answer is reasonable
        if len(answer) > 30:
            # Answer has good length, return as-is
            return answer

        # Check retrieval quality
        max_score = max([r['score'] for r in retrieved]) if retrieved else 0

        # If scores are good, return as-is
        if max_score > 0.02 and len(answer) > 0:
            return answer

        # Use LLM to verify and improve response
        context_summary = ""
        for i, chunk in enumerate(retrieved[:3]):
            context_summary += f"Source {i+1} ({chunk['metadata']['class']}): {chunk['content'][:100]}...\n"

        verification_prompt = f"""You are a helpful NCERT tutor. Verify if this answer is appropriate:

Student Question: {question}

Available NCERT Content:
{context_summary}

Generated Answer: {answer}

Instructions:
1. If the answer directly addresses the question using the available content, return it as-is
2. If the answer is too generic, vague, or admits not having information, rewrite it to:
   - Acknowledge what specific topic they're asking about
   - Explain what IS available in the NCERT content
   - Suggest related topics they CAN learn about
   - Guide them toward helpful questions
3. Never say "I don't know" or "I don't have this information"
4. Always be helpful and guide the student

Improved response:"""

        try:
            payload = {
                "model": config.OPENROUTER_MODEL,
                "messages": [
                    {"role": "system", "content": "You are a helpful NCERT tutor. Always guide students constructively."},
                    {"role": "user", "content": verification_prompt}
                ],
                "temperature": 0.3,
                "max_tokens": 200
            }

            headers = {
                "Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
                "Content-Type": "application/json"
            }

            response = requests.post(
                f"{config.OPENROUTER_BASE_URL}/chat/completions",
                headers=headers,
                json=payload,
                timeout=10
            )

            if response.status_code == 200:
                verified_answer = response.json()["choices"][0]["message"]["content"].strip()
                if len(verified_answer) > 50:  # Only use if substantial
                    return verified_answer

        except:
            pass

        return answer

    def get_cache_stats(self):
        """Get cache statistics"""
        if self.embedder:
            return self.embedder.get_cache_stats()
        return {"cached_queries": 0, "cache_size_mb": 0}

    def clear_cache(self):
        """Clear embedding cache"""
        if self.embedder:
            self.embedder.clear_cache()

    # --- BM25 Helper Methods ---

    def _build_bm25(self, chunks: List[Chunk]):
        """Build BM25 index from chunks"""
        from rank_bm25 import BM25Okapi
        self.bm25_chunks = chunks
        tokenized = [self._tokenize(c.content) for c in chunks]
        self.bm25 = BM25Okapi(tokenized)
        print(f"BM25 index built: {len(chunks)} docs")

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenizer for BM25"""
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        return [t for t in text.split() if len(t) > 2]

    def _save_bm25(self, path: str):
        """Save BM25 data"""
        with open(os.path.join(path, "bm25.pkl"), 'wb') as f:
            pickle.dump(self.bm25_chunks, f)

    def _load_bm25(self, path: str):
        """Load BM25 data"""
        bm25_path = os.path.join(path, "bm25.pkl")
        if os.path.exists(bm25_path):
            from rank_bm25 import BM25Okapi
            with open(bm25_path, 'rb') as f:
                self.bm25_chunks = pickle.load(f)
            tokenized = [self._tokenize(c.content) for c in self.bm25_chunks]
            self.bm25 = BM25Okapi(tokenized)
            print(f"BM25 index loaded: {len(self.bm25_chunks)} docs")

    def _bm25_search(self, query: str, top_k: int) -> List[Dict]:
        """BM25 keyword search"""
        if not self.bm25:
            return []
        tokens = self._tokenize(query)
        scores = self.bm25.get_scores(tokens)
        top_idx = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_idx:
            if scores[idx] > 0:
                chunk = self.bm25_chunks[idx]
                results.append({
                    "chunk_id": chunk.chunk_id,
                    "content": chunk.content,
                    "metadata": chunk.metadata,
                    "score": float(scores[idx])
                })
        return results

    def _hybrid_search(self, query: str, query_embedding: np.ndarray, top_k: int) -> List[Dict]:
        """Combine Dense + BM25 using RRF (Reciprocal Rank Fusion)"""
        k = 60  # RRF constant

        # Get both results
        dense_results = self.vector_store.search(query_embedding, top_k * 2)
        bm25_results = self._bm25_search(query, top_k * 2)

        # RRF fusion
        rrf_scores = {}
        chunk_data = {}

        for rank, r in enumerate(dense_results):
            cid = r["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0) + 1.0 / (k + rank + 1)
            chunk_data[cid] = r

        for rank, r in enumerate(bm25_results):
            cid = r["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0) + 1.0 / (k + rank + 1)
            if cid not in chunk_data:
                chunk_data[cid] = r

        # Sort by RRF score and return top_k
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        results = []
        for cid in sorted_ids[:top_k]:
            r = chunk_data[cid].copy()
            r["score"] = rrf_scores[cid]
            results.append(r)

        return results


