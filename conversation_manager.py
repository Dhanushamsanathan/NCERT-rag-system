"""
Agentic RAG Conversation Manager
===============================
Smart conversation handling with NCERT-aware retrieval
"""

from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class ConversationState:
    """Tracks the current conversation state"""
    current_topic: Optional[str] = None
    topic_chunks: Optional[list] = None
    conversation_turn: int = 0
    last_was_ncert: bool = False


class ConversationManager:
    """Agentic conversation manager with intelligent RAG routing"""

    def __init__(self, rag_pipeline):
        self.rag_pipeline = rag_pipeline
        self.state = ConversationState()
        self.ncert_indicators = [
            "what is", "tell me about", "explain", "describe",
            "according to ncert", "ncert", "class 3", "class 4", "class 5",
            "photosynthesis", "shapes", "collective noun", "grammar",
            "addition", "subtraction", "multiplication", "division",
            "plants", "animals", "environment", "family", "friends",
            "teamwork", "fraction", "geometry", "stories", "poem"
        ]
        self.topic_continuation_words = [
            "how", "why", "when", "where", "which", "who",
            "also", "more", "again", "another", "different",
            "example", "explain", "tell", "show", "can"
        ]

    def needs_ncert_search(self, question: str) -> bool:
        """Determine if question needs NCERT knowledge base"""
        q_lower = question.lower()

        # Check for explicit NCERT indicators
        if any(indicator in q_lower for indicator in self.ncert_indicators):
            return True

        # Check if continuing current topic
        if self._is_continuing_current_topic(question):
            return True

        return False

    def _is_continuing_current_topic(self, question: str) -> bool:
        """Check if question is related to current NCERT topic"""
        if not self.state.current_topic:
            return False

        q_lower = question.lower()

        # Simple keyword-based continuation detection
        if self.state.current_topic.lower() in q_lower:
            return True

        # Check for continuation words
        if any(word in q_lower for word in self.topic_continuation_words):
            if self.state.last_was_ncert:
                return True

        return False

    def route_query(self, question: str) -> Dict[str, any]:
        """Agentic routing based on RAG retrieval results"""
        self.state.conversation_turn += 1
        q_lower = question.lower().strip()

        # Handle explicit conversational queries first
        if any(word in q_lower for word in ["hello", "hi", "hey"]):
            return self._handle_normal_query(question)

        if any(word in q_lower for word in ["thanks", "thank you", "thx"]):
            return self._handle_normal_query(question)

        if any(word in q_lower for word in ["help", "how can you help", "what can you do"]):
            return self._handle_normal_query(question)

        # For everything else, try RAG first and let retrieved content decide
        return self._try_rag_first(question)

    def _try_rag_first(self, question: str) -> Dict[str, any]:
        """Try RAG search first - let retrieved content decide response type"""
        try:
            # Perform RAG search
            result = self.rag_pipeline.query(question, top_k=3)
            sources = result.get('sources', [])

            # Analyze retrieved content
            if self._has_relevant_ncert_content(question, sources):
                # Good NCERT content found - use it
                self.state.last_was_ncert = True

                # Update topic for caching
                if 'topic' in result:
                    self.state.current_topic = result['topic']
                    if result['topic'] in self.rag_pipeline.topic_cache:
                        self.state.topic_chunks = self.rag_pipeline.topic_cache[result['topic']]

                return {
                    **result,
                    "response_type": "ncert_search",
                    "conversation_mode": "ncert_mode"
                }
            else:
                # No relevant content - handle as normal query
                return self._handle_normal_query(question)

        except Exception as e:
            print(f"DEBUG: RAG search failed: {e}")
            return self._handle_normal_query(question)

    def _has_relevant_ncert_content(self, question: str, sources: list) -> bool:
        """Check if retrieved sources are relevant to the question"""
        if not sources:
            return False

        # Check semantic similarity scores
        max_score = max([s['score'] for s in sources[:3]])

        # Quick reject: very low scores
        if max_score < 0.02:
            return False

        # Check content quality and relevance
        total_content_length = sum(len(s['content']) for s in sources[:3])

        # Check if question terms appear in content
        q_words = [w for w in question.lower().split() if len(w) > 3]
        term_matches = 0

        for source in sources[:3]:
            content_lower = source['content'].lower()
            for word in q_words:
                if word in content_lower:
                    term_matches += 1

        # Decision: relevant if good score OR decent score with content matches
        has_content = total_content_length > 200
        has_term_match = term_matches > 0

        return (max_score > 0.1) or (max_score > 0.05 and (has_content or has_term_match))

    def _handle_ncert_query(self, question: str) -> Dict[str, any]:
        """Handle NCERT-related queries with intelligent caching"""
        # Extract topic for caching
        topic = self._extract_topic(question)

        # Check if continuing existing topic
        if self.state.current_topic and self._is_related_to_topic(question, self.state.current_topic):
            return self._handle_topic_continuation(question)

        # New topic - do RAG search
        result = self.rag_pipeline.query(question)

        # Update conversation state
        if 'topic' in result:
            self.state.current_topic = result['topic']
            # Get topic chunks from RAG cache for continuation
            if result['topic'] in self.rag_pipeline.topic_cache:
                self.state.topic_chunks = self.rag_pipeline.topic_cache[result['topic']]

        self.state.last_was_ncert = True

        return {
            **result,
            "response_type": "ncert_search",
            "conversation_mode": "ncert_mode"
        }

    def _handle_topic_continuation(self, question: str) -> Dict[str, any]:
        """Handle follow-up questions about current topic without new RAG search"""
        if not self.state.topic_chunks:
            # Fallback to RAG search if no cached chunks
            return self._handle_ncert_query(question)

        # Use cached chunks for fast response
        context_parts = []
        for i, chunk in enumerate(self.state.topic_chunks[:3]):
            source = f"[{chunk['metadata']['class']}/{chunk['metadata']['subject']}]"
            context_parts.append(f"{source} {chunk['content'][:200]}...")

        context = "\n\n".join(context_parts)

        # Generate contextual answer using cached content
        if self.rag_pipeline.generator:
            try:
                answer = self.rag_pipeline.generator.generate(question, context, self.state.topic_chunks)
            except Exception:
                answer = f"Based on our discussion about {self.state.current_topic}, I found relevant information in the NCERT textbooks."
        else:
            answer = f"I have information about {self.state.current_topic} from the NCERT textbooks we discussed."

        return {
            "question": question,
            "answer": answer,
            "sources": self.state.topic_chunks[:3],
            "response_type": "cached_topic",
            "conversation_mode": "ncert_mode",
            "current_topic": self.state.current_topic,
            "from_cache": True
        }

    def _handle_normal_query(self, question: str) -> Dict[str, any]:
        """Handle normal conversation without RAG search"""
        self.state.last_was_ncert = False

        # Simple conversational responses
        greetings = ["hello", "hi", "hey", "good morning", "good afternoon", "good evening"]
        thanks = ["thanks", "thank you", "thx"]
        help_requests = ["help", "how can you help", "what can you do"]

        q_lower = question.lower()

        # Greeting responses
        if any(greeting in q_lower for greeting in greetings):
            return {
                "question": question,
                "answer": self._generate_greeting_response(),
                "response_type": "conversation",
                "conversation_mode": "normal"
            }

        # Thank you responses
        elif any(thanks_word in q_lower for thanks_word in thanks):
            return {
                "question": question,
                "answer": self._generate_thanks_response(),
                "response_type": "conversation",
                "conversation_mode": "normal"
            }

        # Help requests
        elif any(help_word in q_lower for help_word in help_requests):
            return {
                "question": question,
                "answer": self._generate_help_response(),
                "response_type": "conversation",
                "conversation_mode": "normal"
            }

        # General conversational response
        else:
            return {
                "question": question,
                "answer": self._generate_conversational_response(question),
                "response_type": "conversation",
                "conversation_mode": "normal"
            }

    def _extract_topic(self, question: str) -> str:
        """Extract main topic from question for caching"""
        # Use RAG pipeline's existing topic extraction
        return self.rag_pipeline._extract_topic(question)

    def _is_related_to_topic(self, question: str, topic: str) -> bool:
        """Check if question is related to current topic"""
        q_lower = question.lower()
        topic_lower = topic.lower()

        # Direct topic match
        if topic_lower in q_lower:
            return True

        # Check topic keywords in question
        topic_words = topic_lower.split("_")
        return any(word in q_lower for word in topic_words)

    def _generate_greeting_response(self) -> str:
        """Generate friendly greeting response"""
        greetings = [
            "Hello! 👋 I'm your NCERT tutor helper. I can help you with questions from Class 3-5 textbooks using actual NCERT content.\n\nWhat would you like to learn about today?",
            "Hi there! 📚 I'm here to help with NCERT topics for Classes 3-5. Feel free to ask me questions from your textbooks!\n\nHow can I assist you?",
            "Hello! I'm your friendly NCERT tutor 🌟 I have access to Class 3-5 textbook content and can help explain concepts using examples from your books.\n\nWhat topic would you like to explore?"
        ]
        return greetings[hash(self.state.conversation_turn) % len(greetings)]

    def _generate_thanks_response(self) -> str:
        """Generate polite thank you response"""
        return "You're welcome! 😊 Feel free to ask if you need help with any other NCERT topics!"

    def _generate_help_response(self) -> str:
        """Generate help response"""
        return """I can help you with NCERT textbook content for Classes 3-5! Here's what I can do:

📚 **NCERT Topics**: Math, English, Environmental Studies
🔍 **Smart Search**: Finds relevant information from your textbooks
💡 **Simple Explanations**: Breaks down complex topics in child-friendly ways
📖 **Structured Answers**: Organized with examples and diagrams

**Examples:**
- "What is photosynthesis?"
- "How do bees communicate?"
- "What are collective nouns?"
- "Explain addition with examples"

Just ask your question and I'll help you using actual NCERT content!"""

    def _generate_conversational_response(self, question: str) -> str:
        """Generate general conversational response"""
        return f"I'm here to help with NCERT textbook questions for Classes 1-7! 🎓\n\nCould you ask me about a specific topic from your school books? I can explain concepts using examples directly from the NCERT textbooks.\n\nFor example:\n- \"What is photosynthesis?\"\n- \"What are collective nouns?\"\n- \"Why should we keep our environment clean?\""

    def get_conversation_stats(self) -> Dict[str, any]:
        """Get conversation statistics"""
        return {
            "conversation_turns": self.state.conversation_turn,
            "current_topic": self.state.current_topic,
            "last_query_type": "ncert" if self.state.last_was_ncert else "normal",
            "rag_pipeline_loaded": self.rag_pipeline is not None
        }

    def reset_conversation(self) -> None:
        """Reset conversation state"""
        self.state = ConversationState()