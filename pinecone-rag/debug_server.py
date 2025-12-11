#!/usr/bin/env python3
"""
Debug API Server - Testing without Pinecone initialization
"""

import json
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

class DebugHandler(BaseHTTPRequestHandler):
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
        print(f"GET request for: {path}")

        if path == '/api/health':
            self._json({"status": "ok", "message": "Debug API is running"})
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        path = urlparse(self.path).path
        print(f"POST request for: {path}")

        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        print(f"Received data: {post_data}")

        try:
            data = json.loads(post_data.decode('utf-8'))
            print(f"Parsed data: {data}")
        except:
            print("Failed to parse JSON")
            self._json({"error": "Invalid JSON"}, 400)
            return

        if path == '/api/chat':
            message = data.get('message', '').strip()
            print(f"Message: {message}")

            if not message:
                self._json({"error": "Message is required"}, 400)
                return

            # Simple mock response
            self._json({
                "response": f"Debug: You asked about '{message}'. This is a test response.",
                "sources": [],
                "loading": False
            })

        elif path == '/api/speak':
            text = data.get('text', '')
            print(f"TTS text: {text}")

            if not text:
                self._json({"error": "Text is required"}, 400)
                return

            # Mock TTS response
            self._json({
                "error": "TTS not implemented in debug mode"
            })

        else:
            self.send_error(404)

    def _json(self, data, status=200):
        """Send JSON response"""
        self.send_response(status)
        self._set_cors()
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        response = json.dumps(data)
        self.wfile.write(response.encode())
        print(f"Sent response: {response[:100]}...")

    def log_message(self, format, *args):
        print(f"[{time.strftime('%H:%M:%S')}] {format % args}")

def main():
    print("🔍 Starting Debug API Server...")
    print("📡 Listening on http://localhost:5001")
    print("💡 This version doesn't load Pinecone - for testing only")

    server = HTTPServer(('', 5001), DebugHandler)
    server.serve_forever()

if __name__ == '__main__':
    main()