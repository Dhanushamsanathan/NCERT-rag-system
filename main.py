#!/usr/bin/env python3
"""
NCERT RAG System v1 - Main CLI
==============================
Usage:
    python main.py build     # Build index from NCERT files
    python main.py query     # Interactive Q&A mode
    python main.py ask "your question here"  # Single question
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from rag_pipeline import RAGPipeline
from conversation_manager import ConversationManager


def print_banner():
    """Print welcome banner"""
   
def build_index():
    """Build the RAG index from NCERT documents"""
    print_banner()
    print("Building index from NCERT textbooks...\n")

    rag = RAGPipeline()
    num_chunks = rag.build_index()

    print(f"\nIndex built successfully with {num_chunks} chunks!")
    print(f"Saved to: {config.VECTOR_DB_PATH}/")


# Global Conversation Manager instance for performance
_conversation_manager = None

def get_conversation_manager():
    """Get or create ConversationManager (load once, reuse for all queries)"""
    global _conversation_manager
    if _conversation_manager is None:
        print("Initializing NCERT RAG System...")

        # Initialize RAG Pipeline
        rag = RAGPipeline()

        # Load existing index if available
        if os.path.exists(config.VECTOR_DB_PATH):
            print("Loading index...")
            rag.load_index()
        else:
            print("No index found. Building new index...")
            rag.build_index()

        # Initialize ConversationManager
        _conversation_manager = ConversationManager(rag)
        print("✅ System ready!\n")

    return _conversation_manager

def interactive_mode():
    """Interactive Q&A mode"""
    print_banner()

    # Get ConversationManager (loads only once)
    conv_manager = get_conversation_manager()

    print("\n" + "="*60)
    print("Interactive Q&A Mode")
    print("Type 'quit' or 'exit' to stop")
    print("="*60)

    while True:
        try:
            # Get question
            print()
            question = input("Question: ").strip()

            # Check for exit
            if question.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break

            # Skip empty
            if not question:
                continue

            # Query
            print("\nThinking...")
            result = conv_manager.route_query(question)

            # Display answer
            print("\n" + "-"*40)
            print("ANSWER:")
            print("-"*40)
            print(result['answer'])

            # Show conversation mode
            if 'conversation_mode' in result:
                mode = result['conversation_mode'].upper()
                if mode == "NCERT_MODE":
                    print(f"\n🔍 Mode: {mode}")
                else:
                    print(f"\n💬 Mode: {mode}")

            # Show sources for NCERT queries
            if result.get('conversation_mode') == 'ncert_mode' and 'sources' in result:
                print("\n" + "-"*40)
                print("SOURCES:")
                print("-"*40)
                for i, source in enumerate(result['sources'][:3], 1):
                    meta = source['metadata']
                    score = source['score']
                    print(f"{i}. [{meta['class']}/{meta['subject']}] (score: {score:.3f})")
                print(f"   {source['content'][:100]}...")

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")


def single_query(question: str):
    """Answer a single question"""
    print_banner()

    # Get ConversationManager (loads only once)
    conv_manager = get_conversation_manager()

    # Query
    print(f"\nQuestion: {question}\n")
    print("Thinking...")

    result = conv_manager.route_query(question)

    # Display
    print("\n" + "="*60)
    print("ANSWER:")
    print("="*60)
    print(result['answer'])

    # Show conversation mode
    if 'conversation_mode' in result:
        mode = result['conversation_mode'].upper()
        if mode == "NCERT_MODE":
            print(f"\n🔍 Mode: {mode}")
        else:
            print(f"\n💬 Mode: {mode}")

    # Show sources for NCERT queries only
    if result.get('conversation_mode') == 'ncert_mode' and 'sources' in result:
        print("\n" + "-"*40)
        print("SOURCES:")
        print("-"*40)
        for i, source in enumerate(result['sources'][:3], 1):
            meta = source['metadata']
            print(f"{i}. {meta['class']}/{meta['subject']} (score: {source['score']:.3f})")


def show_help():
    """Show usage help"""
    print("""
NCERT RAG System v1 - Usage
===========================

Commands:
    python main.py build            Build/rebuild the index
    python main.py query            Interactive Q&A mode
    python main.py ask "question"   Ask a single question
    python main.py help             Show this help

Examples:
    python main.py build
    python main.py query
    python main.py ask "What is photosynthesis?"
    python main.py ask "How do plants make food?"
    """)


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        show_help()
        return

    command = sys.argv[1].lower()

    if command == "build":
        build_index()

    elif command == "query":
        interactive_mode()

    elif command == "ask":
        if len(sys.argv) < 3:
            print("Error: Please provide a question")
            print("Usage: python main.py ask \"your question here\"")
            return
        question = " ".join(sys.argv[2:])
        single_query(question)

    elif command in ["help", "-h", "--help"]:
        show_help()

    else:
        print(f"Unknown command: {command}")
        show_help()


if __name__ == "__main__":
    main()
