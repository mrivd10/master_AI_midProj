"""
demo.py - Interactive demo of the RAG system.

Loads the model and index ONCE, then answers questions in a loop. Ideal for a
live demonstration (no per-question reload). Type 'quit' to exit.

Run from the project root with:
    python src/demo.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from rag_system import RAGSystem


def main():
    print("Loading RAG system (one-time)...")
    rag = RAGSystem()
    print("\nReady! Ask a question about Forever Living aloe products.")
    print("Type 'quit' or press Ctrl+C to exit.\n")

    while True:
        try:
            question = input("Question> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break
        if not question:
            continue
        if question.lower() in ("quit", "exit"):
            print("Goodbye.")
            break

        result = rag.answer(question)
        print(f"\nANSWER:\n{result['answer']}")
        print(f"\nSOURCES: {', '.join(result['sources'])}")
        print("-" * 55 + "\n")


if __name__ == "__main__":
    main()
