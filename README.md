# SmartDoc AI – RAG Chatbot with Memory

SmartDoc AI is a Retrieval-Augmented Generation (RAG) based chatbot that allows users to query documents intelligently using LLMs.

## Features
- RAG pipeline using LangChain
- FAISS vector database for semantic search
- Conversational memory for context-aware responses
- Document ingestion and chunking

## Tech Stack
- Python
- LangChain
- OpenAI
- FAISS

## How to Run

1. Install dependencies:
pip install -r requirements.txt

2. Add your OpenAI API key in .env:
OPENAI_API_KEY=your_key_here

3. Run:
python app.py

## Example Use Case
Ask questions about your documents and get intelligent answers with context awareness.

## Future Improvements
- PDF support
- Web UI (Streamlit)
- Multi-document upload
