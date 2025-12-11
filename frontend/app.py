#!/usr/bin/env python3
"""
NCERT RAG Chatbot Web Interface
================================
Simple Flask web server for the chatbot
"""

from flask import Flask, render_template, request, jsonify, Response
from flask_cors import CORS
import sys
import os
import json
import base64
import requests

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_pipeline import RAGPipeline
import re

app = Flask(__name__)
CORS(app)

# Global RAG instance
rag = None

# Conversation cache for follow-up questions
conversation_cache = {
    "last_question": None,
    "last_answer": None,
    "last_topic": None,
    "conversation_history": []
}

# Conversational responses
conversational_responses = {
    "hello": "Hello! 👋 I'm your NCERT tutor helper. I can help you with questions from Classes 1-7 using actual NCERT content.\n\nWhat would you like to learn about today?",
    "hi": "Hi there! 📚 I'm here to help with NCERT topics for Classes 1-7. Feel free to ask me questions from your textbooks!\n\nHow can I assist you?",
    "hey": "Hello! I'm your friendly NCERT tutor 🌟 I have access to Class 1-7 textbook content and can help explain concepts.\n\nWhat topic would you like to explore?",
    "thanks": "You're welcome! 😊 Feel free to ask if you need help with any other NCERT topics!",
    "thank you": "You're welcome! 😊 Feel free to ask if you need help with any other NCERT topics!",
    "thx": "You're welcome! 😊 Feel free to ask if you need help with any other NCERT topics!",
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

def is_follow_up_question(current_question, previous_context):
    """Use LLM to determine if current question is a follow-up"""
    if not previous_context.get('last_question'):
        return False, None

    # Simple LLM call to check if follow-up
    prompt = f"""Previous Question: {previous_context['last_question']}
Previous Answer: {previous_context['last_answer'][:200]}...

Current Question: {current_question}

Is the current question a follow-up or related to the previous question?
Answer with only: YES or NO"""

    try:
        # Use the same RAG generator's LLM
        if rag and rag.generator:
            response = rag.generator.generate(prompt, context="")
            answer = response.strip().upper()
            return answer.startswith("YES"), previous_context['last_topic']
    except:
        pass

    # Fallback keyword-based detection
    q_lower = current_question.lower()
    follow_up_indicators = [
        "more", "explain", "tell me more", "what about", "how about",
        "detail", "further", "elaborate", "continue"
    ]

    # Check if current question contains follow-up indicators
    if any(indicator in q_lower for indicator in follow_up_indicators):
        return True, previous_context['last_topic']

    return False, None

def update_conversation_cache(question, answer):
    """Update the conversation cache with new Q&A"""
    global conversation_cache

    conversation_cache['last_question'] = question
    conversation_cache['last_answer'] = answer

    # Extract topic from answer (simple heuristic)
    if "According to NCERT" in answer:
        lines = answer.split('\n')
        for line in lines[:5]:  # Check first 5 lines
            if '•' in line and (' is ' in line or ' are ' in line):
                # Extract key concept (before 'is' or 'are')
                topic_part = line.split(' is ')[0].split(' are ')[0]
                topic = topic_part.replace('•', '').replace('**', '').strip()
                if len(topic) > 3 and len(topic) < 50:
                    conversation_cache['last_topic'] = topic
                    break

def init_rag():
    """Initialize RAG pipeline"""
    global rag
    if rag is None:
        rag = RAGPipeline()
        # Change to parent directory for vector store
        import os
        original_dir = os.getcwd()
        os.chdir('..')  # Go to parent directory
        rag.load_index()
        os.chdir(original_dir)  # Return to frontend directory
        print("RAG pipeline loaded successfully!")

@app.route('/')
def index():
    """Main chat page"""
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """Chat API endpoint"""
    try:
        data = request.json
        question = data.get('message', '').strip()

        if not question:
            return jsonify({'error': 'No message provided'}), 400

        # Initialize RAG if not done
        init_rag()

        q_lower = question.lower()

        # Check for exact conversational matches
        if q_lower in conversational_responses:
            # Clear cache for new topic
            global conversation_cache
            conversation_cache['last_topic'] = None

            return jsonify({
                'response': conversational_responses[q_lower],
                'mode': 'conversation',
                'sources': []
            })

        # Check if message starts with greetings
        if any(q_lower.startswith(greeting) for greeting in ["hello", "hi", "hey"]):
            # Clear cache for new topic
            conversation_cache['last_topic'] = None

            return jsonify({
                'response': conversational_responses['hello'],
                'mode': 'conversation',
                'sources': []
            })

        # Check if it's a follow-up question
        is_follow_up, last_topic = is_follow_up_question(question, conversation_cache)

        # Build search query
        search_query = question
        if is_follow_up and last_topic:
            # Combine last topic with current question for better RAG results
            search_query = f"{last_topic} {question}"

        # Use RAG for queries
        result = rag.query(search_query, top_k=3)
        sources = result.get('sources', [])

        if sources:
            max_score = max([s['score'] for s in sources])

            if max_score > 0.02:
                # Good content found
                response_text = result['answer']
                mode = 'ncert_rag'

                # Format sources for display
                source_list = []
                for source in sources[:3]:
                    source_list.append({
                        'class': source['metadata']['class'],
                        'subject': source['metadata']['subject'],
                        'score': f"{source['score']:.3f}"
                    })

                # Update conversation cache
                update_conversation_cache(question, response_text)

                return jsonify({
                    'response': response_text,
                    'mode': mode,
                    'sources': source_list
                })
            else:
                # Low score - check if it's a question
                if any(word in q_lower for word in ["who", "what", "when", "where", "why", "how", "explain", "describe"]):
                    response = """I specialize in NCERT textbook content for Classes 1-7 📚

While I don't have information about that specific topic, I can help you with many interesting subjects!

Popular NCERT topics you might like:
🌿 **Science**: Photosynthesis, plants, animals, environment
🔢 **Math**: Addition, subtraction, multiplication, fractions
📖 **English**: Grammar, collective nouns, stories, poems
🏛️ **Social Studies**: Indian history, different types of houses, festivals

Would you like to try asking about any of these NCERT topics?"""

                    return jsonify({
                        'response': response,
                        'mode': 'conversation',
                        'sources': []
                    })
                else:
                    return jsonify({
                        'response': "I can help you with NCERT textbook questions for Classes 1-7! 🎓\n\nTry asking about specific topics from your school books.",
                        'mode': 'conversation',
                        'sources': []
                    })
        else:
            # No sources found
            return jsonify({
                'response': "I can help you with NCERT textbook questions for Classes 1-7! 🎓\n\nTry asking about specific topics from your school books.",
                'mode': 'conversation',
                'sources': []
            })

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

# TTS Configuration
GEMINI_API_KEY = "AIzaSyBpFM3I-RS0irMdu-yXaT5OWtcuE9PaKv0"
GEMINI_TTS_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent"

@app.route('/api/tts/synthesize', methods=['POST'])
def synthesize_speech():
    """Synthesize speech using Gemini TTS API"""
    try:
        data = request.json
        text = data.get('text', '')
        language = data.get('language', 'en')

        if not text:
            return jsonify({'error': 'No text provided'}), 400

        # Language voice mapping for Gemini
        voice_map = {
            'en': 'en-US-Standard-A',
            'ta': 'ta-IN-Standard-A',
            'te': 'te-IN-Standard-A',
            'hi': 'hi-IN-Standard-A'
        }

        voice = voice_map.get(language, 'en-US-Standard-A')

        # Call Gemini TTS API
        payload = {
            "contents": [{
                "parts": [{
                    "text": f"Speak this text naturally: {text}"
                }]
            }],
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "speechConfig": {
                    "voiceConfig": {
                        "prebuiltVoiceConfig": {
                            "voiceName": voice
                        }
                    }
                }
            }
        }

        response = requests.post(
            f"{GEMINI_TTS_URL}?key={GEMINI_API_KEY}",
            json=payload,
            headers={'Content-Type': 'application/json'}
        )

        if response.status_code != 200:
            print(f"TTS API Error: {response.status_code} - {response.text}")
            # Fallback: return success but no audio
            return jsonify({
                'success': False,
                'error': 'TTS service unavailable',
                'fallback': True
            })

        # Extract audio data
        result = response.json()
        if 'candidates' in result and len(result['candidates']) > 0:
            audio_data = result['candidates'][0]['content']['parts'][0]['inlineData']['data']
            return jsonify({
                'success': True,
                'audioData': audio_data
            })
        else:
            return jsonify({
                'success': False,
                'error': 'No audio generated'
            })

    except Exception as e:
        print(f"TTS Error: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'fallback': True
        })

@app.route('/api/tts/synthesize-stream', methods=['POST'])
def synthesize_speech_stream():
    """Synthesize speech with streaming for lower latency"""
    def generate():
        try:
            data = request.json
            text = data.get('text', '')
            language = data.get('language', 'en')

            # Send initial response
            yield f"data: {json.dumps({'type': 'start', 'message': 'Starting synthesis'})}\n\n"

            # For now, use non-streaming approach (simplified)
            # In production, this would use actual streaming from Gemini
            voice_map = {
                'en': 'en-US-Standard-A',
                'ta': 'ta-IN-Standard-A',
                'te': 'te-IN-Standard-A',
                'hi': 'hi-IN-Standard-A'
            }

            voice = voice_map.get(language, 'en-US-Standard-A')

            payload = {
                "contents": [{
                    "parts": [{
                        "text": f"Speak this text naturally: {text}"
                    }]
                }],
                "generationConfig": {
                    "responseModalities": ["AUDIO"],
                    "speechConfig": {
                        "voiceConfig": {
                            "prebuiltVoiceConfig": {
                                "voiceName": voice
                            }
                        }
                    }
                }
            }

            response = requests.post(
                f"{GEMINI_TTS_URL}?key={GEMINI_API_KEY}",
                json=payload,
                headers={'Content-Type': 'application/json'}
            )

            if response.status_code == 200:
                result = response.json()
                if 'candidates' in result and len(result['candidates']) > 0:
                    audio_data = result['candidates'][0]['content']['parts'][0]['inlineData']['data']

                    # Send audio as a chunk
                    yield f"data: {json.dumps({'type': 'chunk', 'data': audio_data, 'chunkNumber': 1})}\n\n"

                    # Send completion
                    yield f"data: {json.dumps({'type': 'done', 'totalChunks': 1, 'timeToFirstAudioMs': 100})}\n\n"
                else:
                    yield f"data: {json.dumps({'type': 'error', 'message': 'No audio generated'})}\n\n"
            else:
                yield f"data: {json.dumps({'type': 'error', 'message': f'TTS API error: {response.status_code}'})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return Response(generate(), mimetype='text/event-stream')

if __name__ == '__main__':
    init_rag()
    app.run(debug=True, host='0.0.0.0', port=5001)