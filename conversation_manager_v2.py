"""
Agentic RAG Conversation Manager v2
==================================
Intelligent hybrid routing with fast checks, search-first approach, and confidence-based decisions
"""

from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
import re


@dataclass
class ConversationState:
    """Tracks the current conversation state"""
    current_topic: Optional[str] = None
    topic_chunks: Optional[list] = None
    conversation_turn: int = 0
    last_was_ncert: bool = False


class ConversationManagerV2:
    """Intelligent conversation manager with hybrid routing strategy"""

    def __init__(self, rag_pipeline):
        self.rag_pipeline = rag_pipeline
        self.state = ConversationState()

        # Thresholds for decision making
        self.CONFIDENCE_THRESHOLD = 0.02  # Minimum score to consider content relevant
        self.HIGH_CONFIDENCE_THRESHOLD = 0.1  # Score for very relevant content

        # Quick match patterns for fast routing
        self.greeting_patterns = [
            "hello", "hi", "hey", "good morning", "good afternoon",
            "good evening", "how are you", "what's up"
        ]

        self.thanks_patterns = [
            "thanks", "thank you", "thx", "appreciate"
        ]

        self.help_patterns = [
            "help", "how can you help", "what can you do", "what do you do"
        ]

        # Math problem indicators
        self.math_patterns = [
            r"how many", r"how much", r"calculate", r"what is.*\d+[\+\-\*\/]\s*\d+",
            r"what is .* -", r"what is .* ×", r"what is .* ÷",
            r"costs? ₹", r"costs? \$", r"total", r"sum", r"add"
        ]

        # Academic query indicators
        self.academic_indicators = [
            "what is", "what are", "explain", "describe", "why", "how",
            "when", "where", "which", "who", "define", "tell me about",
            "photosynthesis", "collective noun", "grammar", "plants",
            "animals", "environment", "shapes", "house", "water"
        ]

    def route_query(self, question: str) -> Dict[str, any]:
        """Intelligent routing with agentic decision making"""
        self.state.conversation_turn += 1
        q_lower = question.lower().strip()

        # Layer 1: Only handle explicit conversational queries
        if self._is_greeting(q_lower):
            return self._handle_greeting()

        if self._is_thanks(q_lower):
            return self._handle_thanks()

        if self._is_help_request(q_lower):
            return self._handle_help()

        # Layer 2: Use RAG for everything else
        # Let the RAG system decide if content is relevant
        return self._try_rag_first(question)

    def _is_greeting(self, question: str) -> bool:
        """Check if it's a greeting"""
        return any(pattern in question for pattern in self.greeting_patterns)

    def _is_thanks(self, question: str) -> bool:
        """Check if it's a thanks"""
        return any(pattern in question for pattern in self.thanks_patterns)

    def _is_help_request(self, question: str) -> bool:
        """Check if it's a help request"""
        return any(pattern in question for pattern in self.help_patterns)

    def _is_math_problem(self, question: str) -> bool:
        """Check if it's a math problem"""
        # First check for simple arithmetic with numbers
        if re.search(r'\b\d+[\+\-\*\/]\s*\d+', question):
            return True

        # Then check for other math patterns
        for pattern in self.math_patterns:
            if re.search(pattern, question):
                return True
        return False

    def _is_academic_query(self, question: str) -> bool:
        """Check if it's an academic query"""
        # Quick academic indicator check
        if any(indicator in question for indicator in self.academic_indicators):
            return True

        # Check for question words
        question_words = ["what", "why", "how", "when", "where", "which", "who", "explain", "describe"]
        has_question_word = any(question.startswith(word) for word in question_words)

        # If it starts with a question word, likely academic
        if has_question_word:
            return True

        # Check if it mentions NCERT topics
        ncert_topics = [
            "photosynthesis", "collective noun", "grammar", "plants", "animals",
            "environment", "family", "shapes", "house", "water", "addition",
            "subtraction", "multiplication", "division", "fraction"
        ]
        return any(topic in question for topic in ncert_topics)

    def _handle_greeting(self) -> Dict[str, any]:
        """Handle greeting with varied responses"""
        greetings = [
            "Hello! 👋 I'm your NCERT tutor helper. I can help you with questions from Classes 1-7 using actual NCERT content.\n\nWhat would you like to learn about today?",
            "Hi there! 📚 I'm here to help with NCERT topics for Classes 1-7. Feel free to ask me questions from your textbooks!\n\nHow can I assist you?",
            "Hello! I'm your friendly NCERT tutor 🌟 I have access to Class 1-7 textbook content and can help explain concepts using examples from your books.\n\nWhat topic would you like to explore?"
        ]

        return {
            "question": "",
            "answer": greetings[self.state.conversation_turn % len(greetings)],
            "response_type": "conversation",
            "conversation_mode": "normal"
        }

    def _handle_thanks(self) -> Dict[str, any]:
        """Handle thanks with polite response"""
        return {
            "question": "",
            "answer": "You're welcome! 😊 Feel free to ask if you need help with any other NCERT topics!",
            "response_type": "conversation",
            "conversation_mode": "normal"
        }

    def _handle_help(self) -> Dict[str, any]:
        """Handle help requests with system capabilities"""
        help_text = """I can help you with NCERT textbook content for Classes 1-7! Here's what I can do:

📚 **NCERT Topics**: Math, Science, English, Social Studies
🔍 **Smart Search**: Finds relevant information from your textbooks
💡 **Simple Explanations**: Breaks down complex topics in child-friendly ways
📖 **Structured Answers**: Organized with examples and diagrams

**Examples of questions I can answer:**
- "What is photosynthesis?"
- "Why should we keep our environment clean?"
- "What are collective nouns?"
- "A pencil costs ₹5. How much do 4 pencils cost?"
- "What are different types of houses?"

Just ask your question and I'll help you using actual NCERT content!"""

        return {
            "question": "",
            "answer": help_text,
            "response_type": "conversation",
            "conversation_mode": "normal"
        }

    def _handle_math_problem(self, question: str) -> Dict[str, any]:
        """Handle math problems with RAG for examples"""
        # Try to find similar math problems in NCERT
        try:
            # Search for math content
            math_keywords = ["math", "arithmetic", "addition", "subtraction", "multiply", "cost", "price"]
            refined_query = question
            for keyword in math_keywords:
                if keyword not in question.lower():
                    refined_query = f"{refined_query} {keyword}"

            result = self.rag_pipeline.query(refined_query, top_k=3)

            # Check if we found relevant math content
            max_score = max([s['score'] for s in result.get('sources', [])]) if result.get('sources') else 0

            if max_score > self.CONFIDENCE_THRESHOLD:
                # Format answer with math example
                return {
                    "question": question,
                    "answer": self._format_math_answer(question, result),
                    "sources": result.get('sources', [])[:3],
                    "response_type": "math_rag",
                    "conversation_mode": "ncert_mode"
                }
        except:
            pass

        # Fallback: Try to solve directly
        return self._solve_math_direct(question)

    def _handle_academic_query(self, question: str) -> Dict[str, any]:
        """Handle academic queries with RAG"""
        # Check if continuing existing topic
        if self.state.current_topic and self._is_continuing_topic(question):
            return self._handle_topic_continuation(question)

        # Fresh RAG search
        return self._perform_rag_search(question)

    def _try_rag_first(self, question: str) -> Dict[str, any]:
        """Agentic RAG-only approach - let retrieved content decide the response"""
        try:
            # Perform RAG search
            result = self.rag_pipeline.query(question, top_k=3)
            sources = result.get('sources', [])

            # Analyze retrieved content to determine response type
            content_analysis = self._analyze_retrieved_content(question, sources)

            if content_analysis['has_relevant_content']:
                # Content found - use RAG result
                return {
                    **result,
                    "response_type": content_analysis['response_type'],
                    "conversation_mode": "ncert_mode",
                    "content_analysis": content_analysis
                }
            else:
                # No relevant content found
                return self._try_direct_llm(question)

        except Exception as e:
            print(f"DEBUG: RAG search failed: {e}")
            return self._try_direct_llm(question)

    def _analyze_retrieved_content(self, question: str, sources: List[Dict]) -> Dict[str, any]:
        """Intelligently analyze retrieved content to decide response type"""
        if not sources:
            return {
                "has_relevant_content": False,
                "response_type": "direct",
                "confidence": 0.0,
                "analysis": "No sources found"
            }

        # Check semantic relevance scores
        scores = [s['score'] for s in sources]
        max_score = max(scores)
        avg_score = sum(scores) / len(scores)

        # Analyze content characteristics
        total_content = ""
        content_samples = []

        for source in sources[:3]:
            content = source.get('content', '')
            total_content += content + " "
            if len(content) > 50:
                content_samples.append(content[:200])

        # Check content quality indicators
        content_length = len(total_content)

        # Determine relevance based on multiple factors
        relevance_factors = {
            "semantic_similarity": max_score > self.CONFIDENCE_THRESHOLD,
            "content_volume": content_length > 300,
            "has_structured_content": any('•' in c or ':' in c for c in content_samples),
            "question_terms_present": any(
                word.lower() in total_content.lower()
                for word in question.split()
                if len(word) > 3
            )
        ]

        # Count positive factors
        positive_factors = sum(relevance_factors.values())

        # Decision logic
        has_relevant = (
            max_score > self.HIGH_CONFIDENCE_THRESHOLD or  # Strong semantic match
            (max_score > self.CONFIDENCE_THRESHOLD and positive_factors >= 2) or  # Good match + other factors
            (avg_score > self.CONFIDENCE_THRESHOLD and content_length > 500)  # Decent average + substantial content
        )

        # Determine response type based on content analysis
        response_type = "rag"  # Default
        if has_relevant:
            # Check if content contains arithmetic/math operations
            has_arithmetic = any(
                op in total_content
                for op in ['+', '-', '×', '÷', '=', 'calculate', 'cost', 'price']
            )
            if has_arithmetic and any(char.isdigit() for char in question):
                response_type = "math_rag"
            else:
                response_type = "ncert_search"

        return {
            "has_relevant_content": has_relevant,
            "response_type": response_type,
            "confidence": max_score,
            "analysis": {
                "max_score": max_score,
                "avg_score": avg_score,
                "content_length": content_length,
                "positive_factors": positive_factors,
                "factors": relevance_factors
            }
        }

    def _has_relevant_content(self, sources: List[Dict]) -> bool:
        """Legacy method - use _analyze_retrieved_content instead"""
        analysis = self._analyze_retrieved_content("", sources)
        return analysis['has_relevant_content']

    def _is_continuing_topic(self, question: str) -> bool:
        """Check if continuing current topic"""
        if not self.state.current_topic:
            return False

        # Simple keyword check
        topic_words = self.state.current_topic.lower().replace('_', ' ').split()
        return any(word in question.lower() for word in topic_words)

    def _handle_topic_continuation(self, question: str) -> Dict[str, any]:
        """Handle follow-up questions using cached content"""
        if self.state.topic_chunks:
            return {
                "question": question,
                "answer": self._generate_structured_answer(question, self.state.topic_chunks),
                "sources": self.state.topic_chunks[:3],
                "response_type": "cached_topic",
                "conversation_mode": "ncert_mode",
                "from_cache": True
            }

        # No cached content - search again
        return self._perform_rag_search(question)

    def _perform_rag_search(self, question: str) -> Dict[str, any]:
        """Perform RAG search with error handling"""
        try:
            result = self.rag_pipeline.query(question)

            # Update conversation state
            if 'topic' in result:
                self.state.current_topic = result['topic']
                if result['topic'] in self.rag_pipeline.topic_cache:
                    self.state.topic_chunks = self.rag_pipeline.topic_cache[result['topic']]

            self.state.last_was_ncert = True

            # Check if answer is too short
            if len(result.get('answer', '').strip()) < 30:
                result['answer'] = self._generate_fallback_answer(question, result.get('sources', []))

            return {
                **result,
                "response_type": "ncert_search",
                "conversation_mode": "ncert_mode"
            }

        except Exception as e:
            print(f"DEBUG: RAG search error: {e}")
            return self._try_direct_llm(question)

    def _try_direct_llm(self, question: str) -> Dict[str, any]:
        """Try answering without RAG"""
        # For now, return a helpful message encouraging NCERT questions
        return {
            "question": question,
            "answer": "I can help you with NCERT textbook questions for Classes 1-7! 🎓\n\nTry asking about specific topics from your school books. For example:\n- \"What is photosynthesis?\"\n- \"What are collective nouns?\"\n- \"Why should we keep our environment clean?\"\n- \"What are different types of houses?\"",
            "response_type": "direct",
            "conversation_mode": "normal"
        }

    def _format_math_answer(self, question: str, result: Dict) -> str:
        """Format math answer with examples from NCERT"""
        sources = result.get('sources', [])
        answer_parts = []

        # Try to solve the specific problem
        solution = self._solve_math_direct(question)
        if "₹" in solution or "=" in solution:
            answer_parts.append(solution)

        # Add NCERT example
        if sources:
            answer_parts.append("\n**According to NCERT** 📚")
            answer_parts.append("Here's a similar example from your textbook:")

            for source in sources[:2]:
                if 'cost' in source['content'].lower() or 'price' in source['content'].lower():
                    # Extract relevant part
                    content = source['content'][:200]
                    answer_parts.append(f"• From {source['metadata']['class']}: {content}...")
                    break

        return "\n".join(answer_parts)

    def _solve_math_direct(self, question: str) -> str:
        """Attempt to solve math problems directly"""
        # Extract numbers and operations
        q_lower = question.lower()

        # Handle simple arithmetic first
        arithmetic_match = re.search(r'(\d+)\s*([\+\-\*\/])\s*(\d+)', question)
        if arithmetic_match:
            num1 = int(arithmetic_match.group(1))
            op = arithmetic_match.group(2)
            num2 = int(arithmetic_match.group(3))

            if op == '+':
                result = num1 + num2
                operation = "add"
            elif op == '-':
                result = num1 - num2
                operation = "subtract"
            elif op == '*':
                result = num1 * num2
                operation = "multiply"
            elif op == '/':
                if num2 != 0:
                    result = num1 / num2
                    operation = "divide"
                else:
                    return "Cannot divide by zero!"

            return f"The answer is: {num1} {op} {num2} = {result}"

        # Handle "A costs X, how much for Y" pattern
        cost_pattern = r"(\w+)\s+costs?\s+₹(\d+).*?(\d+)\s+\1"
        match = re.search(cost_pattern, q_lower)

        if match:
            item = match.group(1)
            cost_per_item = int(match.group(2))
            quantity = int(match.group(3))
            total = cost_per_item * quantity

            return f"**Solution:**\n• Cost of 1 {item} = ₹{cost_per_item}\n• Cost of {quantity} {item}s = {quantity} × ₹{cost_per_item} = ₹{total}\n\n**Answer:** {quantity} {item}s cost ₹{total}"

        # Return generic math response
        return "I can help with math problems! Please provide the numbers clearly. For example: \"What is 5 + 3?\" or \"A pencil costs ₹5. How much do 4 pencils cost?\""

    def _generate_fallback_answer(self, question: str, sources: List[Dict]) -> str:
        """Generate structured answer when LLM fails"""
        if sources:
            return f"**According to NCERT** 📚\n\nI found information about this topic in your textbooks. Let me help you understand:\n\n• Based on the NCERT content, {question.lower()} is an important concept.\n• You can find more details in your class {sources[0]['metadata']['class']} textbook."

        return "I can help you with NCERT textbook questions! Could you rephrase your question or ask about a specific topic from Classes 1-7?"

    def _generate_structured_answer(self, question: str, sources: List[Dict]) -> str:
        """Generate structured NCERT answer from cached sources"""
        if not sources:
            return self._generate_fallback_answer(question, [])

        # Extract key information from sources
        key_points = []

        for source in sources[:3]:
            content = source['content']
            meta = source['metadata']

            # Look for relevant sentences
            sentences = content.split('. ')
            for sentence in sentences[:3]:
                if len(sentence) > 20:
                    key_points.append(f"• {sentence.strip()}.")

        # Build answer
        answer = f"**According to NCERT** 📚\n\n"
        if key_points:
            answer += "\n".join(key_points[:3])
        else:
            answer += f"• This topic is covered in your {meta['class']} {meta['subject']} textbook."

        answer += f"\n\n**Try this:** Can you find examples of this in your textbook? 📚"

        return answer

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