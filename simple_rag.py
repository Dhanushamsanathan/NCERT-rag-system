#!/usr/bin/env python3
"""
Simple RAG CLI - Direct RAG retrieval approach
==============================================
Usage:
    python simple_rag.py "your question here"
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rag_pipeline import RAGPipeline


def main():
    if len(sys.argv) < 2:
        print("Usage: python simple_rag.py \"your question\"")
        sys.exit(1)

    question = " ".join(sys.argv[1:])
    q_lower = question.lower().strip()

    # Quick check for conversational messages - no RAG needed
    conversational_responses = {
        # Greetings
        "hello": "Hello! 👋 I'm your NCERT tutor helper. I can help you with questions from Classes 1-7 using actual NCERT content.\n\nWhat would you like to learn about today?",
        "hi": "Hi there! 📚 I'm here to help with NCERT topics for Classes 1-7. Feel free to ask me questions from your textbooks!\n\nHow can I assist you?",
        "hey": "Hello! I'm your friendly NCERT tutor 🌟 I have access to Class 1-7 textbook content and can help explain concepts.\n\nWhat topic would you like to explore?",
        # Thanks
        "thanks": "You're welcome! 😊 Feel free to ask if you need help with any other NCERT topics!",
        "thank you": "You're welcome! 😊 Feel free to ask if you need help with any other NCERT topics!",
        "thx": "You're welcome! 😊 Feel free to ask if you need help with any other NCERT topics!",
        # Help
        "help": """I can help you with NCERT textbook content for Classes 1-7! Here's what I can do:

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
    }

    # Check for exact conversational matches
    if q_lower in conversational_responses:
        print("\n" + "="*50)
        print(f"QUESTION: {question}")
        print("="*50)
        print(f"\nMODE: CONVERSATION 💬")
        print(f"\n{conversational_responses[q_lower]}")
        return

    # Check if message starts with greetings
    if any(q_lower.startswith(greeting) for greeting in ["hello", "hi", "hey"]):
        print("\n" + "="*50)
        print(f"QUESTION: {question}")
        print("="*50)
        print(f"\nMODE: CONVERSATION 💬")
        print(f"\n{conversational_responses['hello']}")
        return

    # Initialize RAG pipeline only for non-conversational queries
    rag = RAGPipeline()
    rag.load_index()
    result = rag.query(question, top_k=3)

    # Print results
    print("\n" + "="*50)
    print(f"QUESTION: {question}")
    print("="*50)

    if result.get('sources'):
        max_score = max([s['score'] for s in result['sources']])
        print(f"\nDEBUG: Max score = {max_score}")
        if max_score > 0.02:
            # Good content found
            print(f"\nMODE: NCERT RAG 🔍")
            print(f"\nANSWER:\n{result['answer']}")

            print(f"\nSOURCES:")
            for i, source in enumerate(result['sources'][:3], 1):
                print(f"\n{i}. [{source['metadata']['class']}] {source['metadata']['subject']} - Score: {source['score']:.3f}")
                print(f"   {source['content'][:150]}...")
        else:
            # Low score content
            print(f"\nMODE: CONVERSATION 💬 (Low similarity: {max_score:.3f})")
            print(f"\nI can help you with NCERT textbook questions for Classes 1-7! 🎓")
    else:
        # No good content found
        print(f"\nMODE: CONVERSATION 💬 (No sources found)")
        print(f"\nI can help you with NCERT textbook questions for Classes 1-7! 🎓")
        print(f"\nTry asking about specific topics from your school books:")
        print(f"  - \"What is photosynthesis?\"")
        print(f"  - \"What are collective nouns?\"")
        print(f"  - \"Why should we keep our environment clean?\"")
        print(f"  - \"What are different types of houses?\"")


if __name__ == "__main__":
    main()