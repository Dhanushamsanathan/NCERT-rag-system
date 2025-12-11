// Import existing services
import { PineconeRAGService } from '../../pinecone-rag/services/pineconeRAGService.js';
import { GeminiSpeechService } from '../../pinecone-rag/services/geminiSpeechService.js';

// Global variables
let currentAudio = null;
let isRecording = false;
let recognition = null;
let ragService = null;
let speechService = null;
let isLoading = false;

// Initialize services
async function initializeServices() {
    try {
        // Initialize RAG service
        ragService = new PineconeRAGService('http://localhost:5001/api');

        // Initialize speech service
        speechService = new GeminiSpeechService({
            apiKey: 'AIzaSyBpFM3I-RS0irMdu-yXaT5OWtcuE9PaKv0',
            language: 'en-IN-Wavenet-D',
            baseUrl: 'http://localhost:5001/api'
        });

        // Initialize speech recognition
        if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
            const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
            recognition = new SpeechRecognition();
            recognition.continuous = false;
            recognition.interimResults = false;
            recognition.lang = 'en-IN';

            recognition.onresult = (event) => {
                const transcript = event.results[0][0].transcript;
                document.getElementById('messageInput').value = transcript;
                stopRecording();
                sendMessage();
            };

            recognition.onerror = (event) => {
                console.error('Speech recognition error:', event.error);
                stopRecording();
                showNotification('Speech recognition failed. Please try again.');
            };

            recognition.onend = () => {
                stopRecording();
            };
        }

        console.log('Services initialized successfully');
    } catch (error) {
        console.error('Error initializing services:', error);
        showNotification('Failed to initialize services');
    }
}

// DOM elements
const chatMessages = document.getElementById('chatMessages');
const messageInput = document.getElementById('messageInput');
const sendBtn = document.getElementById('sendBtn');
const micBtn = document.getElementById('micBtn');
const loader = document.getElementById('loader');
const connectionStatus = document.getElementById('connectionStatus');

// Event listeners
messageInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

// Functions
function askExample(question) {
    messageInput.value = question;
    sendMessage();
}

async function sendMessage() {
    const message = messageInput.value.trim();
    if (!message || isLoading) return;

    // Add user message
    addMessage(message, 'user');
    messageInput.value = '';
    isLoading = true;
    updateSendButton();

    // Show typing indicator
    showTypingIndicator();

    try {
        // Query RAG service
        const response = await ragService.query({ question: message });

        hideTypingIndicator();

        if (response.answer) {
            // Add bot response
            addMessage(response.answer, 'bot', response.sources);

            // Auto-play TTS
            setTimeout(() => {
                const lastBotMessage = document.querySelector('.message:last-child .btn-audio');
                if (lastBotMessage) {
                    lastBotMessage.click();
                }
            }, 500);
        } else {
            addMessage("I couldn't find information about that in NCERT books. Try asking about photosynthesis, water cycle, or grammar!", 'bot');
        }
    } catch (error) {
        hideTypingIndicator();
        console.error('Error:', error);
        addMessage("Sorry, I'm having trouble connecting. Please try again.", 'bot');
    }

    isLoading = false;
    updateSendButton();
}

function addMessage(text, sender, sources = []) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}`;

    const avatar = document.createElement('div');
    avatar.className = 'avatar';
    avatar.textContent = sender === 'user' ? '👤' : '🤖';

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';

    const textDiv = document.createElement('div');
    textDiv.className = 'text';
    textDiv.innerHTML = text;

    contentDiv.appendChild(textDiv);

    // Add audio button for bot messages
    if (sender === 'bot') {
        const actionsDiv = document.createElement('div');
        actionsDiv.className = 'actions';

        const audioBtn = document.createElement('button');
        audioBtn.className = 'btn-audio';
        audioBtn.innerHTML = '<i class="icon">🔊</i> Play';
        audioBtn.onclick = () => playMessage(audioBtn, text);

        actionsDiv.appendChild(audioBtn);
        contentDiv.appendChild(actionsDiv);
    }

    // Add sources if available
    if (sources && sources.length > 0) {
        const sourcesDiv = document.createElement('div');
        sourcesDiv.className = 'sources';
        sourcesDiv.style.marginTop = '8px';
        sourcesDiv.style.fontSize = '12px';
        sourcesDiv.style.color = '#666';
        sourcesDiv.innerHTML = `<i>📚 Sources: ${sources.map(s => s.metadata.class + ' - ' + s.metadata.subject).join(', ')}</i>`;
        contentDiv.appendChild(sourcesDiv);
    }

    messageDiv.appendChild(avatar);
    messageDiv.appendChild(contentDiv);

    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

async function playMessage(button, text) {
    // Stop current audio if playing
    if (currentAudio) {
        currentAudio.pause();
        currentAudio = null;
        document.querySelectorAll('.btn-audio').forEach(btn => {
            btn.classList.remove('playing');
            btn.innerHTML = '<i class="icon">🔊</i> Play';
        });
        return;
    }

    // Mark as playing
    button.classList.add('playing');
    button.innerHTML = '<i class="icon">⏸️</i> Pause';

    try {
        // Use speech service or fallback API
        const response = await fetch('http://localhost:5001/api/speak', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                text: text,
                language: 'en'
            })
        });

        const data = await response.json();

        if (data.audio) {
            currentAudio = new Audio('data:audio/mp3;base64,' + data.audio);
            currentAudio.onended = () => {
                button.classList.remove('playing');
                button.innerHTML = '<i class="icon">🔊</i> Play';
                currentAudio = null;
            };
            currentAudio.play();
        } else {
            throw new Error(data.error || 'No audio received');
        }
    } catch (error) {
        console.error('Speech error:', error);
        button.classList.remove('playing');
        button.innerHTML = '<i class="icon">🔊</i> Play';

        if (error.error === 'quota_exceeded') {
            showNotification('Speech quota exceeded. Text-only mode available.');
        } else {
            showNotification('Speech temporarily unavailable');
        }
    }
}

function toggleRecording() {
    if (!recognition) {
        showNotification('Speech recognition not supported in your browser');
        return;
    }

    if (isRecording) {
        stopRecording();
    } else {
        startRecording();
    }
}

function startRecording() {
    isRecording = true;
    micBtn.classList.add('recording');
    micBtn.innerHTML = '<i class="icon">⏹️</i>';
    recognition.start();
}

function stopRecording() {
    isRecording = false;
    micBtn.classList.remove('recording');
    micBtn.innerHTML = '<i class="icon">🎤</i>';
    if (recognition) {
        recognition.stop();
    }
}

function showTypingIndicator() {
    const typingDiv = document.createElement('div');
    typingDiv.className = 'message bot';
    typingDiv.id = 'typing-indicator';
    typingDiv.innerHTML = `
        <div class="avatar">🤖</div>
        <div class="message-content">
            <div class="text">
                <div class="typing-indicator">
                    <span class="typing-dot"></span>
                    <span class="typing-dot"></span>
                    <span class="typing-dot"></span>
                </div>
            </div>
        </div>
    `;
    chatMessages.appendChild(typingDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function hideTypingIndicator() {
    const typingIndicator = document.getElementById('typing-indicator');
    if (typingIndicator) {
        typingIndicator.remove();
    }
}

function updateSendButton() {
    sendBtn.disabled = isLoading;
    if (isLoading) {
        sendBtn.innerHTML = '<i class="icon">⏳</i>';
    } else {
        sendBtn.innerHTML = '<i class="icon">➤</i>';
    }
}

function showNotification(message) {
    // Create a simple notification
    const notification = document.createElement('div');
    notification.style.cssText = `
        position: fixed;
        bottom: 20px;
        left: 50%;
        transform: translateX(-50%);
        background: #333;
        color: white;
        padding: 12px 24px;
        border-radius: 8px;
        z-index: 1000;
        animation: fadeIn 0.3s ease;
    `;
    notification.textContent = message;
    document.body.appendChild(notification);

    setTimeout(() => {
        notification.remove();
    }, 3000);
}

// Initialize on load
document.addEventListener('DOMContentLoaded', () => {
    initializeServices();
});

// Check connection
setInterval(() => {
    fetch('http://localhost:5001/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: 'ping' })
    })
    .then(() => {
        connectionStatus.innerHTML = '<span class="status-dot"></span><span>Connected</span>';
    })
    .catch(() => {
        connectionStatus.innerHTML = '<span class="status-dot" style="background: #ff6b6b;"></span><span>Disconnected</span>';
    });
}, 5000);