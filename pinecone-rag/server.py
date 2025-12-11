#!/usr/bin/env python3
"""
Simple NCERT RAG Server with Pinecone
========================================
Uses Pinecone for fast search and Gemini for speech
"""

import os
import json
import base64
import requests
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from rag_pipeline_pinecone import PineconeRAGPipeline

# API Keys
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set")
GEMINI_TTS_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent"

# Initialize RAG pipeline globally
rag_pipeline = PineconeRAGPipeline()
rag_pipeline.load_index()
print("✅ Pinecone RAG ready!")

class SimpleNCERTHandler(BaseHTTPRequestHandler):

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', '*')
        self.send_header('Access-Control-Allow-Headers', '*')
        self.end_headers()

    def do_POST(self):
        path = urlparse(self.path).path
        content_length = int(self.headers['Content-Length'])

        data = json.loads(self.rfile.read(content_length).decode('utf-8'))

        if path == '/api/chat':
            self._chat(data)
        elif path == '/api/speak':
            self._speak(data)
        else:
            self.send_error(404)

    def do_GET(self):
        path = urlparse(self.path).path

        if path == '/':
            self._serve_html()
        else:
            self.send_error(404)

    def _chat(self, data):
        """Chat with NCERT content"""
        message = data.get('message', '')

        if not message:
            self._json({"error": "No message"}, 400)
            return

        # Simple responses
        greetings = {
            "hello": "Hello! I'm your NCERT helper. What can I help you with?",
            "hi": "Hi! Ask me about anything from Classes 1-7!",
            "thanks": "You're welcome! Feel free to ask more questions."
        }

        if message.lower() in greetings:
            self._json({"response": greetings[message.lower()]})
            return

        # Search NCERT content
        result = rag_pipeline.query(message)

        if result['answer']:
            self._json({
                "response": result['answer'],
                "sources": result.get('sources', [])
            })
        else:
            self._json({
                "response": "I couldn't find that in NCERT books. Try asking about photosynthesis, collective nouns, or environment!"
            })

    def _speak(self, data):
        """Text to speech"""
        text = data.get('text', '')

        if not text:
            self._json({"error": "No text"}, 400)
            return

        # Indian voices
        voices = {
            'en': 'en-IN-Wavenet-D',
            'hi': 'hi-IN-Wavenet-D',
            'ta': 'ta-IN-Wavenet-A',
            'te': 'te-IN-Wavenet-A'
        }

        voice = voices.get(data.get('language', 'en'), 'en-IN-Wavenet-D')

        # Call Gemini TTS
        payload = {
            "contents": [{
                "parts": [{
                    "text": f"Speak like a friendly Indian teacher: {text}"
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

        try:
            response = requests.post(
                f"{GEMINI_TTS_URL}?key={GEMINI_API_KEY}",
                json=payload
            )

            if response.status_code == 200:
                result = response.json()
                audio = result['candidates'][0]['content']['parts'][0]['inlineData']['data']
                self._json({"audio": audio})
            else:
                self._json({"error": "Speech failed"})
        except:
            self._json({"error": "Speech service unavailable"})

    def _serve_html(self):
        """Serve main HTML file"""
        html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>📚 NCERT Learning Assistant</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }

        .chat-container {
            width: 100%;
            max-width: 900px;
            height: 90vh;
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }

        .chat-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px 30px;
            display: flex;
            align-items: center;
            gap: 15px;
        }

        .chat-header h1 {
            font-size: 1.5rem;
            font-weight: 600;
        }

        .chat-header .subtitle {
            font-size: 0.9rem;
            opacity: 0.9;
            margin-top: 5px;
        }

        .chat-messages {
            flex: 1;
            padding: 30px;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 20px;
        }

        .chat-messages::-webkit-scrollbar {
            width: 8px;
        }

        .chat-messages::-webkit-scrollbar-track {
            background: #f1f1f1;
        }

        .chat-messages::-webkit-scrollbar-thumb {
            background: #888;
            border-radius: 4px;
        }

        .message {
            display: flex;
            gap: 12px;
            animation: fadeIn 0.3s ease;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .message.user {
            flex-direction: row-reverse;
        }

        .message-avatar {
            width: 40px;
            height: 40px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.2rem;
            flex-shrink: 0;
        }

        .message.user .message-avatar {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }

        .message.bot .message-avatar {
            background: #f0f0f0;
            color: #666;
        }

        .message-content {
            max-width: 70%;
        }

        .message-bubble {
            padding: 15px 20px;
            border-radius: 20px;
            word-wrap: break-word;
            position: relative;
        }

        .message.user .message-bubble {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-bottom-right-radius: 5px;
        }

        .message.bot .message-bubble {
            background: #f0f0f0;
            color: #333;
            border-bottom-left-radius: 5px;
        }

        .message-time {
            font-size: 0.75rem;
            color: #999;
            margin-top: 5px;
            text-align: right;
        }

        .message.bot .message-time {
            text-align: left;
        }

        .audio-controls {
            display: flex;
            gap: 10px;
            margin-top: 10px;
        }

        .audio-btn {
            background: none;
            border: none;
            color: #666;
            cursor: pointer;
            font-size: 1.2rem;
            padding: 8px 12px;
            border-radius: 50%;
            transition: all 0.3s ease;
        }

        .audio-btn:hover {
            background: #e0e0e0;
        }

        .audio-btn.playing {
            color: #667eea;
        }

        .typing-indicator {
            display: none;
            align-items: center;
            gap: 8px;
            padding: 15px 20px;
            background: #f0f0f0;
            border-radius: 20px;
            width: fit-content;
        }

        .typing-indicator span {
            width: 8px;
            height: 8px;
            background: #666;
            border-radius: 50%;
            animation: typing 1.4s infinite;
        }

        .typing-indicator span:nth-child(2) { animation-delay: 0.2s; }
        .typing-indicator span:nth-child(3) { animation-delay: 0.4s; }

        @keyframes typing {
            0%, 60%, 100% { transform: translateY(0); }
            30% { transform: translateY(-10px); }
        }

        .chat-input-container {
            padding: 20px 30px;
            background: #f8f9fa;
            border-top: 1px solid #e0e0e0;
        }

        .chat-input-wrapper {
            display: flex;
            gap: 15px;
            align-items: flex-end;
        }

        .chat-input {
            flex: 1;
            background: white;
            border: 2px solid #e0e0e0;
            border-radius: 25px;
            padding: 15px 20px;
            font-size: 1rem;
            resize: none;
            max-height: 120px;
            transition: border-color 0.3s ease;
        }

        .chat-input:focus {
            outline: none;
            border-color: #667eea;
        }

        .send-btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            transition: all 0.3s ease;
        }

        .send-btn:hover {
            transform: scale(1.1);
        }

        .send-btn:disabled {
            background: #ccc;
            cursor: not-allowed;
        }

        .examples {
            display: flex;
            gap: 10px;
            margin-bottom: 15px;
            flex-wrap: wrap;
        }

        .example-chip {
            background: white;
            border: 1px solid #e0e0e0;
            border-radius: 20px;
            padding: 8px 15px;
            font-size: 0.9rem;
            cursor: pointer;
            transition: all 0.3s ease;
        }

        .example-chip:hover {
            background: #667eea;
            color: white;
            border-color: #667eea;
        }
    </style>
</head>
<body>
    <div class="chat-container">
        <div class="chat-header">
            <i class="fas fa-graduation-cap" style="font-size: 2rem;"></i>
            <div>
                <h1>NCERT Learning Assistant</h1>
                <div class="subtitle">Your personal tutor for Classes 1-7 📚</div>
            </div>
        </div>

        <div class="chat-messages" id="chatMessages">
            <div class="message bot">
                <div class="message-avatar">🤖</div>
                <div class="message-content">
                    <div class="message-bubble">
                        Hello! 👋 I'm your NCERT Learning Assistant. I can help you with all subjects from Classes 1-7. What would you like to learn about today?
                    </div>
                    <div class="message-time">Just now</div>
                </div>
            </div>
        </div>

        <div class="typing-indicator" id="typingIndicator">
            <span></span>
            <span></span>
            <span></span>
        </div>

        <div class="chat-input-container">
            <div class="examples">
                <div class="example-chip" onclick="askExample('What is photosynthesis?')">What is photosynthesis?</div>
                <div class="example-chip" onclick="askExample('Tell me about the solar system')">Tell me about the solar system</div>
                <div class="example-chip" onclick="askExample('What are collective nouns?')">What are collective nouns?</div>
                <div class="example-chip" onclick="askExample('How do plants grow?')">How do plants grow?</div>
            </div>
            <div class="chat-input-wrapper">
                <textarea
                    id="messageInput"
                    class="chat-input"
                    placeholder="Ask me anything about NCERT topics..."
                    rows="1"
                ></textarea>
                <button id="sendBtn" class="send-btn" onclick="sendMessage()">
                    <i class="fas fa-paper-plane"></i>
                </button>
            </div>
        </div>
    </div>

    <script>
        let isPlaying = false;
        let currentAudio = null;

        // Auto-resize textarea
        const messageInput = document.getElementById('messageInput');
        messageInput.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = Math.min(this.scrollHeight, 120) + 'px';
        });

        function askExample(question) {
            messageInput.value = question;
            sendMessage();
        }

        function addMessage(content, isUser = false) {
            const messagesContainer = document.getElementById('chatMessages');
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${isUser ? 'user' : 'bot'}`;

            const time = new Date().toLocaleTimeString('en-US', {
                hour: 'numeric',
                minute: '2-digit'
            });

            messageDiv.innerHTML = `
                <div class="message-avatar">${isUser ? '<i class="fas fa-user"></i>' : '🤖'}</div>
                <div class="message-content">
                    <div class="message-bubble">${content}</div>
                    <div class="message-time">${time}</div>
                    ${!isUser ? `
                        <div class="audio-controls">
                            <button class="audio-btn" onclick="toggleAudio('${content.replace(/'/g, "\\'")}')" title="Play/Pause">
                                <i class="fas fa-play"></i>
                            </button>
                        </div>
                    ` : ''}
                </div>
            `;

            messagesContainer.appendChild(messageDiv);
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }

        function showTypingIndicator() {
            document.getElementById('typingIndicator').style.display = 'flex';
            const messagesContainer = document.getElementById('chatMessages');
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }

        function hideTypingIndicator() {
            document.getElementById('typingIndicator').style.display = 'none';
        }

        async function sendMessage() {
            const text = messageInput.value.trim();
            if (!text) return;

            addMessage(text, true);
            messageInput.value = '';
            messageInput.style.height = 'auto';

            const sendBtn = document.getElementById('sendBtn');
            sendBtn.disabled = true;
            sendBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i>';

            showTypingIndicator();

            try {
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({message: text})
                });

                const data = await response.json();
                hideTypingIndicator();
                addMessage(data.response || 'Sorry, I couldn\'t understand that. Could you rephrase it?');
            } catch (error) {
                hideTypingIndicator();
                addMessage('Error: ' + error.message);
            }

            sendBtn.disabled = false;
            sendBtn.innerHTML = '<i class="fas fa-paper-plane"></i>';
        }

        async function toggleAudio(text) {
            if (isPlaying && currentAudio) {
                currentAudio.pause();
                isPlaying = false;
                document.querySelectorAll('.audio-btn i').forEach(icon => {
                    icon.className = 'fas fa-play';
                });
                return;
            }

            // Update all audio buttons to show loading
            document.querySelectorAll('.audio-btn i').forEach(icon => {
                icon.className = 'fas fa-spinner fa-spin';
            });

            try {
                const response = await fetch('/api/speak', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({text: text})
                });

                const data = await response.json();
                if (data.audio) {
                    currentAudio = new Audio('data:audio/mp3;base64,' + data.audio);
                    currentAudio.onplay = () => {
                        isPlaying = true;
                        document.querySelectorAll('.audio-btn i').forEach(icon => {
                            icon.className = 'fas fa-pause';
                        });
                    };
                    currentAudio.onended = () => {
                        isPlaying = false;
                        document.querySelectorAll('.audio-btn i').forEach(icon => {
                            icon.className = 'fas fa-play';
                        });
                    };
                    currentAudio.play();
                }
            } catch (error) {
                console.error('Audio error:', error);
                document.querySelectorAll('.audio-btn i').forEach(icon => {
                    icon.className = 'fas fa-play';
                });
            }
        }

        messageInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });
    </script>
</body>
</html>
        """

        self.send_response(200)
        self.send_header('Content-Type', 'text/html')
        self.end_headers()
        self.wfile.write(html.encode())

    def _json(self, data, status=200):
        """Send JSON response"""
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def send_error(self, code, message="Not found"):
        self.send_response(code)
        self.end_headers()

def main():
    print("🌟 Starting NCERT Helper Server...")
    server = HTTPServer(('', 5001), SimpleNCERTHandler)
    print("📚 Pinecone RAG loaded")
    print("🎤 Gemini TTS ready")
    print("🌐 Server: http://localhost:5001")
    server.serve_forever()

if __name__ == '__main__':
    main()