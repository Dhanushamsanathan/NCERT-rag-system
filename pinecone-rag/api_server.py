#!/usr/bin/env python3
"""
NCERT API Server
===============
Minimal Python server with API endpoints for TypeScript frontend
"""

import os
import json
import sys
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag_pipeline_pinecone import PineconeRAGPipeline

# Global variables
rag_pipeline = None
pipeline_lock = threading.Lock()
pipeline_thread = None

class APIHandler(BaseHTTPRequestHandler):
    def _set_cors(self):
        """Set CORS headers"""
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')

    def do_OPTIONS(self):
        self.send_response(200)
        self._set_cors()
        self.end_headers()

    def do_GET(self):
        path = urlparse(self.path).path

        # Simple health check
        if path == '/api/health':
            self._json({"status": "ok", "message": "NCERT API is running"})
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        global rag_pipeline

        path = urlparse(self.path).path
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        data = json.loads(post_data.decode('utf-8'))

        if path == '/api/chat':
            # Chat endpoint using Pinecone RAG
            message = data.get('message', '').strip()

            if not message:
                self._json({"error": "Message is required"}, 400)
                return

            # Initialize pipeline if needed
            with pipeline_lock:
                if rag_pipeline is None:
                    # Load in background and return loading message
                    global pipeline_thread
                    if pipeline_thread is None:
                        pipeline_thread = threading.Thread(target=self._load_pipeline, daemon=True)
                        pipeline_thread.start()
                    self._json({
                        "response": "🔍 Loading NCERT database... Please wait.",
                        "loading": True
                    })
                    return

                # Check if still loading
                if rag_pipeline == 'loading':
                    self._json({
                        "response": "Still loading... Please wait.",
                        "loading": True
                    })
                    return
                elif rag_pipeline == 'error':
                    self._json({
                        "error": "Failed to load NCERT database. Please check server logs.",
                        "loading": False
                    }, 500)
                    return

            # Query RAG system
            try:
                result = rag_pipeline.query(message)

                if result['answer']:
                    self._json({
                        "response": result['answer'],
                        "sources": result.get('sources', []),
                        "loading": False
                    })
                else:
                    self._json({
                        "response": "I couldn't find that in NCERT books. Try asking about photosynthesis, water cycle, or grammar!",
                        "loading": False
                    })
            except Exception as e:
                self._json({"error": str(e), "loading": False}, 500)

        elif path == '/api/speak':
            # TTS endpoint using Gemini
            text = data.get('text', '')
            language = data.get('language', 'en')

            if not text:
                self._json({"error": "Text is required"}, 400)
                return

            try:
                import requests
                GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
                if not GEMINI_API_KEY:
                    raise ValueError("GEMINI_API_KEY environment variable not set")
                url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent"

                # Natural Indian teacher prompt
                voices = {
                    'en': 'en-IN-Wavenet-D',  # Indian English male
                    'hi': 'hi-IN-Wavenet-D',  # Hindi
                    'female': 'en-IN-Wavenet-A'  # Indian English female
                }

                voice = voices.get(language, 'en-IN-Wavenet-D')

                payload = {
                    "contents": [{
                        "parts": [{
                            "text": f"You are a warm, friendly Indian teacher. Explain this concept clearly with encouraging words: {text}"
                        }]
                    }],
                    "generationConfig": {
                        "responseModalities": ["AUDIO"],
                        "speechConfig": {
                            "voiceConfig": {
                                "prebuiltVoiceConfig": {"voiceName": voice}
                            }
                        }
                    }
                }

                response = requests.post(
                    f"{url}?key={GEMINI_API_KEY}",
                    json=payload,
                    timeout=15
                )

                if response.status_code == 200:
                    result = response.json()
                    audio_data = result['candidates'][0]['content']['parts'][0]['inlineData']['data']
                    self._json({
                        "audio": audio_data,
                        "voice": voice
                    })
                else:
                    # Return error with quota info
                    self._json({
                        "error": "Speech service unavailable. Please try text-only mode.",
                        "quota_exceeded": True
                    })
            except Exception as e:
                self._json({"error": "Speech service error"}, 500)

        else:
            self.send_error(404)

    def _load_pipeline(self):
        """Load RAG pipeline in background"""
        global rag_pipeline
        with pipeline_lock:
            rag_pipeline = 'loading'  # Mark as loading

        try:
            rag = PineconeRAGPipeline()
            rag.load_index()
            with pipeline_lock:
                rag_pipeline = rag
            print("✅ Pinecone RAG loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading pipeline: {e}")
            import traceback
            traceback.print_exc()
            with pipeline_lock:
                rag_pipeline = 'error'

    def _json(self, data, status=200):
        """Send JSON response"""
        self.send_response(status)
        self._set_cors()
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def log_message(self, format, *args):
        pass  # Disable logging

def main():
    print("🚀 Starting NCERT API Server...")
    print("🌐 API Endpoints:")
    print("   POST /api/chat - Chat with NCERT content")
    print("   POST /api/speak - Text to speech")
    print("📚 Pinecone RAG will load on first request")
    print("\n💡 Frontend should run on http://localhost:3000")
    print("🔗 API available at http://localhost:5001")

    server = HTTPServer(('', 5001), APIHandler)
    server.serve_forever()

if __name__ == '__main__':
    main()