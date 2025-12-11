# NCERT RAG Chatbot Frontend

A simple web interface for the NCERT RAG system with real-time chat capabilities.

## Features

- 🎨 **Clean, Kid-Friendly UI**: Simple chat interface designed for students
- ⚡ **Real-time Responses**: Instant chat with typing indicators
- 📱 **Mobile Responsive**: Works on phones, tablets, and computers
- 💬 **Smart Conversation Handling**: Quick responses for greetings
- 📚 **NCERT Attribution**: Shows sources for academic answers
- 🎯 **Topic Suggestions**: Helps users when queries are out of scope

## Installation

1. Install dependencies:
```bash
cd frontend
pip install -r requirements.txt
```

2. Run the web server:
```bash
python app.py
```

3. Open your browser and go to:
```
http://localhost:5000
```

## How It Works

### The Chat Interface
- **Blue bubbles**: Your messages
- **Gray bubbles**: Chatbot responses
- **Sources**: Shows NCERT book and class for academic answers

### Response Types

1. **Greetings** (Instant): hello, hi, hey, thanks, help
2. **NCERT Questions**: Full textbook content with diagrams and examples
3. **Out of Scope**: Helpful suggestions for related NCERT topics

### Example Interactions

Try these questions:
- "What is photosynthesis?"
- "Explain collective nouns"
- "Why should we keep our environment clean?"
- "A pencil costs ₹5, how much do 4 pencils cost?"

## API Endpoint

The chat uses a simple REST API:

```http
POST /api/chat
Content-Type: application/json

{
  "message": "What is photosynthesis?"
}
```

Response:
```json
{
  "response": "**According to NCERT** 📚...",
  "mode": "ncert_rag",
  "sources": [
    {
      "class": "class-7",
      "subject": "science",
      "score": "0.033"
    }
  ]
}
```

## Architecture

```
Frontend (HTML/JS) → Flask API → simple_rag.py → rag_pipeline.py → NCERT Content
```

The frontend communicates with the Flask backend, which uses the same logic as `simple_rag.py` to process queries.

## Development

To run in debug mode with auto-reload:
```bash
FLASK_ENV=development python app.py
```

## Production Deployment

For production, use a proper WSGI server like Gunicorn:
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```