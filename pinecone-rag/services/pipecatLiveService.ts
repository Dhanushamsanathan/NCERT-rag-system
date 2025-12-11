/**
 * Pipecat Live Voice Service - Daily Transport Only
 *
 * Provides real-time voice interaction with Gemini Live model through Pipecat.
 * Uses Daily.co transport for all environments (local WSL2 and production).
 *
 * Connection Flow:
 * 1. Frontend calls POST /api/connect to create Daily room and get credentials
 * 2. Bot joins the Daily room on backend
 * 3. Frontend connects to the same Daily room
 * 4. Audio streams bidirectionally through Daily's infrastructure
 */

import { PipecatClient } from '@pipecat-ai/client-js';
import type { PipecatTransport } from '@pipecat-ai/client-js';
import { DailyTransport } from '@pipecat-ai/daily-transport';

export interface PipecatLiveConfig {
  /**
   * Base URL for the Pipecat service (bot_runner.py)
   * Default: from runtime config AI_SERVICE_URL
   */
  baseUrl?: string;

  /**
   * Language code for the conversation
   * Default: 'ta' (Tamil)
   */
  language?: string;

  /**
   * Enable microphone for voice input
   * Default: true
   */
  enableMicrophone?: boolean;

  /**
   * Enable camera for video
   * Default: false
   */
  enableCamera?: boolean;

  /**
   * Enable automatic reconnection on disconnect
   * Default: true
   */
  autoReconnect?: boolean;

  /**
   * Maximum reconnection attempts
   * Default: 3
   */
  maxReconnectAttempts?: number;

  /**
   * Reconnection delay in ms
   * Default: 2000
   */
  reconnectDelay?: number;

  /**
   * Learning context to send to the bot
   */
  learningContext?: {
    curriculumTitle?: string;
    curriculumLevel?: string;
    moduleTitle?: string;
    moduleDescription?: string;
    modulePrompt?: string;
    roleplayPrompt?: string;
    isRoleplayMode?: boolean;
    isReturningSession?: boolean;
    conversationHistory?: Array<{ role: string; content: string }>;
    studyMaterials?: Array<{ title: string; content?: string; url?: string }>;
    // Roleplay scenario selection properties
    roleplayCharacter?: string;  // Character name for display
    roleplayVoice?: string;      // Voice ID: 'Puck' (male) or 'Charon' (female)
    roleplayGender?: 'male' | 'female';  // Gender for voice selection fallback
    selectedScenarioId?: number | null;  // ID of selected scenario (null = default)
    selectedScenarioTitle?: string;      // Title of selected scenario for logging
  };
}

export interface PipecatLiveCallbacks {
  onBotStartedSpeaking?: () => void;
  onBotStoppedSpeaking?: () => void;
  onUserStartedSpeaking?: () => void;
  onUserStoppedSpeaking?: () => void;
  onUserTranscript?: (transcript: string, isFinal: boolean) => void;
  onBotTranscript?: (transcript: string, isFinal: boolean) => void;
  onConnectionStateChange?: (state: string) => void;
  onError?: (error: Error) => void;
  onConnected?: () => void;
  onDisconnected?: () => void;
  onReconnecting?: (attempt: number, maxAttempts: number) => void;
  onReconnected?: () => void;
  onReconnectFailed?: () => void;
  onBotResponseTimeout?: () => void;
  onInitialGreetingTimeout?: () => void;
  onGreetingRetryFailed?: () => void;
  onRoomCreationFailed?: () => void;
}

/**
 * Live voice service using Pipecat with Daily.co transport
 */
class PipecatLiveService {
  private client: PipecatClient | null = null;
  private transport: PipecatTransport | null = null;
  private isConnected = false;
  private isBotSpeaking = false;
  private isUserSpeaking = false;
  private config: Required<PipecatLiveConfig>;
  private callbacks: PipecatLiveCallbacks = {};
  private isMicrophoneActive = false;

  // Daily room info for cleanup
  private currentRoomUrl: string | null = null;
  private currentRoomName: string | null = null;

  // Transcription state
  private currentUserTranscript: string = '';
  private finalUserTranscript: string = '';
  private transcriptionResolve: ((transcript: string) => void) | null = null;
  private transcriptionTimeout: ReturnType<typeof setTimeout> | null = null;
  private isWaitingForTranscription: boolean = false;

  // Push-to-talk state
  private isPushToTalkActive: boolean = false;
  private pushToTalkStartTime: number = 0;

  // Reconnection state
  private reconnectAttempts: number = 0;
  private isReconnecting: boolean = false;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;

  // Connection monitoring
  private heartbeatInterval: ReturnType<typeof setInterval> | null = null;
  private lastActivityTime: number = Date.now();
  private connectionHealthy: boolean = true;

  // Bot response timeout
  private botResponseTimeout: ReturnType<typeof setTimeout> | null = null;
  private isWaitingForBotResponse: boolean = false;
  private botResponseTimeoutMs: number = 10000;

  // Initial greeting timeout
  private initialGreetingTimeout: ReturnType<typeof setTimeout> | null = null;
  private hasReceivedInitialGreeting: boolean = false;
  private initialGreetingTimeoutMs: number = 15000;
  private greetingRetryAttempts: number = 0;
  private maxGreetingRetryAttempts: number = 2;

  constructor() {
    this.config = {
      baseUrl: useRuntimeConfig().public.AI_SERVICE_URL,
      language: 'ta',
      enableMicrophone: true,
      enableCamera: false,
      autoReconnect: true,
      maxReconnectAttempts: 3,
      reconnectDelay: 2000,
    } as Required<PipecatLiveConfig>;
  }

  /**
   * Check if WebRTC is supported in the browser
   */
  isSupported(): boolean {
    return !!(
      window.RTCPeerConnection &&
      navigator.mediaDevices &&
      navigator.mediaDevices.getUserMedia
    );
  }

  /**
   * Connect to the Pipecat bot via Daily.co transport
   */
  async connect(
    config: Partial<PipecatLiveConfig> = {},
    callbacks: PipecatLiveCallbacks = {}
  ): Promise<void> {
    if (this.isConnected) {
      console.warn('Already connected to Pipecat service');
      return;
    }

    if (!this.isSupported()) {
      throw new Error('WebRTC is not supported in this browser');
    }

    this.config = { ...this.config, ...config } as Required<PipecatLiveConfig>;
    this.callbacks = callbacks;

    try {
      console.log('🔗 Connecting to Pipecat service via Daily transport');
      console.log('📍 Base URL:', this.config.baseUrl);

      // Request Daily room from backend
      console.log('📞 Requesting Daily room from backend...');

      const connectResponse = await fetch(`${this.config.baseUrl}/api/connect`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ learningContext: this.config.learningContext || {} })
      });

      if (!connectResponse.ok) {
        const errorText = await connectResponse.text();
        console.error('❌ Failed to create Daily room:', errorText);

        if (this.callbacks.onRoomCreationFailed) {
          this.callbacks.onRoomCreationFailed();
        }

        throw new Error(`Failed to create Daily room: ${connectResponse.status} - ${errorText}`);
      }

      const roomData = await connectResponse.json();
      console.log('✅ Daily room created:', roomData.room_url);
      console.log('🏷️ Room name:', roomData.room_name);

      this.currentRoomUrl = roomData.room_url;
      this.currentRoomName = roomData.room_name;

      // Create Daily transport with callbacks in constructor
      console.log('🚀 Creating Daily transport...');
      this.transport = new DailyTransport();

      // Prepare connect options
      // For public rooms (test mode), token may be null
      const connectOptions: { url: string; token?: string } = {
        url: roomData.room_url
      };
      if (roomData.token) {
        connectOptions.token = roomData.token;
        console.log('🔑 Using meeting token for private room');
      } else {
        console.log('🔓 No token required (public room / test mode)');
      }

      // Create Pipecat client with callbacks
      this.client = new PipecatClient({
        transport: this.transport,
        enableMic: this.config.enableMicrophone,
        enableCam: this.config.enableCamera,
        callbacks: {
          onConnected: () => {
            console.log('✅ PipecatClient onConnected callback fired');
          },
          onDisconnected: () => {
            console.log('📴 PipecatClient onDisconnected callback fired');
          },
          onBotReady: () => {
            console.log('🤖 Bot is ready');
          }
        }
      });

      console.log('🎤 PipecatClient created with Daily transport');

      // Set up event listeners BEFORE connecting
      this.setupEventListeners();

      // Connect to Daily room
      console.log('🔌 Connecting to Daily room...');
      console.log('📍 URL:', connectOptions.url);

      try {
        await this.client.connect(connectOptions);
        console.log('✅ client.connect() resolved successfully');
      } catch (connectError) {
        console.error('❌ client.connect() threw error:', connectError);
        // Check if we're actually connected despite the error
        // Some errors are non-fatal warnings
        if (this.transport && (this.transport as any).state === 'connected') {
          console.log('⚠️ Error occurred but transport appears connected, continuing...');
        } else {
          throw connectError;
        }
      }

      // Finalize connection
      await new Promise(resolve => setTimeout(resolve, 1000));

      this.isConnected = true;
      this.connectionHealthy = true;
      this.lastActivityTime = Date.now();
      this.reconnectAttempts = 0;
      this.hasReceivedInitialGreeting = false;

      console.log('✅ Successfully connected to Pipecat service');

      // Send learning context to the bot via app message
      // The bot waits for this before starting the conversation
      if (this.config.learningContext) {
        console.log('📚 Sending learning context to bot...');
        this.sendLearningContext(this.config.learningContext);
      }

      this.startHeartbeat();
      this.startInitialGreetingTimeout();

      if (this.callbacks.onConnected) {
        this.callbacks.onConnected();
      }

    } catch (error) {
      console.error('❌ Error connecting to Pipecat service:', error);
      this.cleanup();

      if (this.callbacks.onError) {
        this.callbacks.onError(error as Error);
      }

      throw error;
    }
  }

  /**
   * Set up event listeners for Pipecat client
   */
  private setupEventListeners(): void {
    if (!this.client) return;

    // Track event - handle remote audio/video tracks
    this.client.on('trackStarted', (track: MediaStreamTrack, participant?: any) => {
      console.log('🎵 Received track:', track.kind, 'from participant:', participant);

      // Only play REMOTE audio tracks (from the bot), not local tracks (our own mic)
      const isLocal = participant?.local === true;
      const isBot = participant?.name === 'AI Mentor' || !isLocal;

      if (track.kind === 'audio' && !isLocal) {
        console.log('🔊 Setting up bot audio playback for:', participant?.name || 'unknown');

        let audioElement = document.getElementById('pipecat-bot-audio') as HTMLAudioElement;

        if (!audioElement) {
          audioElement = document.createElement('audio');
          audioElement.id = 'pipecat-bot-audio';
          audioElement.autoplay = true;
          audioElement.playsInline = true;
          document.body.appendChild(audioElement);
          console.log('✅ Created audio element for bot audio');
        }

        const stream = new MediaStream([track]);
        audioElement.srcObject = stream;
        audioElement.muted = false;
        audioElement.volume = 1.0;

        audioElement.play().then(() => {
          console.log('✅ Bot audio is playing! Volume:', audioElement.volume);
        }).catch(err => {
          console.error('❌ Failed to play bot audio:', err);
          const playOnClick = () => {
            audioElement.play().then(() => {
              console.log('✅ Audio started after user interaction');
              document.removeEventListener('click', playOnClick);
            }).catch(e => console.error('Still cannot play:', e));
          };
          document.addEventListener('click', playOnClick, { once: true });
        });
      } else if (track.kind === 'audio' && isLocal) {
        console.log('🎤 Skipping local audio track (our own mic)');
      }
    });

    // Bot speaking events
    this.client.on('botStartedSpeaking', () => {
      console.log('Bot started speaking');
      this.isBotSpeaking = true;
      this.updateActivity();
      this.clearBotResponseTimeout();

      if (!this.hasReceivedInitialGreeting) {
        console.log('✅ Received initial greeting from bot');
        this.hasReceivedInitialGreeting = true;
        this.greetingRetryAttempts = 0;
        this.clearInitialGreetingTimeout();
      }

      if (this.callbacks.onBotStartedSpeaking) {
        this.callbacks.onBotStartedSpeaking();
      }
    });

    this.client.on('botStoppedSpeaking', () => {
      console.log('Bot stopped speaking');
      this.isBotSpeaking = false;
      this.isWaitingForBotResponse = false;
      this.updateActivity();
      if (this.callbacks.onBotStoppedSpeaking) {
        this.callbacks.onBotStoppedSpeaking();
      }
    });

    // User speaking events
    this.client.on('userStartedSpeaking', () => {
      console.log('User started speaking');
      this.isUserSpeaking = true;
      this.updateActivity();
      if (this.callbacks.onUserStartedSpeaking) {
        this.callbacks.onUserStartedSpeaking();
      }
    });

    this.client.on('userStoppedSpeaking', () => {
      console.log('User stopped speaking');
      this.isUserSpeaking = false;
      this.updateActivity();
      this.signalUserStoppedSpeaking();
      if (this.callbacks.onUserStoppedSpeaking) {
        this.callbacks.onUserStoppedSpeaking();
      }
    });

    // Transcript events
    this.client.on('userTranscript', (data: any) => {
      console.log('🎤 User transcript:', data.text, 'Final:', data.final);

      if (data.final) {
        this.finalUserTranscript = data.text;
        this.sendUserTranscriptionToBackend(data.text);

        if (this.isWaitingForTranscription && this.transcriptionResolve) {
          this.transcriptionResolve(data.text);
          this.transcriptionResolve = null;
          this.isWaitingForTranscription = false;
          if (this.transcriptionTimeout) {
            clearTimeout(this.transcriptionTimeout);
            this.transcriptionTimeout = null;
          }
        }
      } else {
        this.currentUserTranscript = data.text;
      }

      if (this.callbacks.onUserTranscript) {
        this.callbacks.onUserTranscript(data.text, data.final);
      }
    });

    this.client.on('botTranscript', (data: any) => {
      if (this.callbacks.onBotTranscript) {
        this.callbacks.onBotTranscript(data.text, data.final);
      }
    });

    // Connection events
    this.client.on('connectionStateChange', (state: string) => {
      console.log('📡 Connection state:', state);
      if (this.callbacks.onConnectionStateChange) {
        this.callbacks.onConnectionStateChange(state);
      }
    });

    this.client.on('error', (error: Error) => {
      console.error('❌ Pipecat error:', error);
      if (this.callbacks.onError) {
        this.callbacks.onError(error);
      }
    });

    this.client.on('disconnected', () => {
      console.log('📡 Disconnected from Daily room');
      const wasConnected = this.isConnected;
      this.isConnected = false;
      this.connectionHealthy = false;
      this.stopHeartbeat();

      if (this.callbacks.onDisconnected) {
        this.callbacks.onDisconnected();
      }

      const timeSinceActivity = Date.now() - this.lastActivityTime;
      const wasIdle = timeSinceActivity > 120000;

      if (wasConnected && this.config.autoReconnect && !this.isReconnecting && !wasIdle) {
        console.log('🔄 Attempting auto-reconnect...');
        this.attemptReconnect();
      } else {
        this.cleanup();
      }
    });
  }

  private startHeartbeat(): void {
    this.stopHeartbeat();
    this.heartbeatInterval = setInterval(() => {
      const timeSinceActivity = Date.now() - this.lastActivityTime;
      if (timeSinceActivity > 120000 && this.isConnected) {
        console.warn(`⚠️ No activity for ${Math.round(timeSinceActivity / 1000)}s`);
      }
    }, 30000);
  }

  private stopHeartbeat(): void {
    if (this.heartbeatInterval) {
      clearInterval(this.heartbeatInterval);
      this.heartbeatInterval = null;
    }
  }

  private startBotResponseTimeout(): void {
    this.clearBotResponseTimeout();
    this.isWaitingForBotResponse = true;
    this.botResponseTimeout = setTimeout(() => {
      if (this.isWaitingForBotResponse && this.isConnected) {
        console.warn('⚠️ Bot response timeout');
        this.isWaitingForBotResponse = false;
        if (this.callbacks.onBotResponseTimeout) {
          this.callbacks.onBotResponseTimeout();
        }
      }
    }, this.botResponseTimeoutMs);
  }

  private clearBotResponseTimeout(): void {
    if (this.botResponseTimeout) {
      clearTimeout(this.botResponseTimeout);
      this.botResponseTimeout = null;
    }
    this.isWaitingForBotResponse = false;
  }

  private startInitialGreetingTimeout(): void {
    this.clearInitialGreetingTimeout();
    console.log(`⏱️ Greeting timeout started (attempt ${this.greetingRetryAttempts + 1}/${this.maxGreetingRetryAttempts + 1})`);

    this.initialGreetingTimeout = setTimeout(() => {
      if (!this.hasReceivedInitialGreeting && this.isConnected) {
        console.warn('⚠️ Bot greeting timeout');

        if (this.greetingRetryAttempts >= this.maxGreetingRetryAttempts) {
          if (this.callbacks.onGreetingRetryFailed) {
            this.callbacks.onGreetingRetryFailed();
          }
          return;
        }

        this.greetingRetryAttempts++;
        if (this.callbacks.onInitialGreetingTimeout) {
          this.callbacks.onInitialGreetingTimeout();
        }
      }
    }, this.initialGreetingTimeoutMs);
  }

  private clearInitialGreetingTimeout(): void {
    if (this.initialGreetingTimeout) {
      clearTimeout(this.initialGreetingTimeout);
      this.initialGreetingTimeout = null;
    }
  }

  private updateActivity(): void {
    this.lastActivityTime = Date.now();
  }

  private async attemptReconnect(): Promise<void> {
    if (this.isReconnecting || this.reconnectAttempts >= this.config.maxReconnectAttempts) {
      if (this.reconnectAttempts >= this.config.maxReconnectAttempts) {
        console.error('❌ Max reconnection attempts reached');
        this.cleanup();
        if (this.callbacks.onReconnectFailed) {
          this.callbacks.onReconnectFailed();
        }
      }
      return;
    }

    this.isReconnecting = true;
    this.reconnectAttempts++;

    console.log(`🔄 Reconnect attempt ${this.reconnectAttempts}/${this.config.maxReconnectAttempts}`);

    if (this.callbacks.onReconnecting) {
      this.callbacks.onReconnecting(this.reconnectAttempts, this.config.maxReconnectAttempts);
    }

    this.cleanup();
    await new Promise(resolve => setTimeout(resolve, this.config.reconnectDelay));

    try {
      await this.connect(this.config, this.callbacks);
      console.log('✅ Reconnected!');
      this.reconnectAttempts = 0;
      this.isReconnecting = false;
      if (this.callbacks.onReconnected) {
        this.callbacks.onReconnected();
      }
    } catch (error) {
      console.error('❌ Reconnect failed:', error);
      this.isReconnecting = false;
      this.reconnectTimer = setTimeout(() => this.attemptReconnect(), this.config.reconnectDelay);
    }
  }

  async disconnect(disableAutoReconnect: boolean = true): Promise<void> {
    if (!this.isConnected || !this.client) return;

    if (disableAutoReconnect) {
      this.config.autoReconnect = false;
    }

    try {
      console.log('Disconnecting from Daily room');
      this.stopHeartbeat();

      if (this.currentRoomName) {
        try {
          await fetch(`${this.config.baseUrl}/api/disconnect`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ room_name: this.currentRoomName })
          });
          console.log('✅ Room cleanup requested');
        } catch (e) {
          console.warn('⚠️ Room cleanup failed:', e);
        }
      }

      await this.client.disconnect();
      this.cleanup();
      this.greetingRetryAttempts = 0;
    } catch (error) {
      console.error('Error disconnecting:', error);
      this.cleanup();
      throw error;
    }
  }

  private cleanup(): void {
    this.isConnected = false;
    this.isBotSpeaking = false;
    this.isUserSpeaking = false;
    this.connectionHealthy = false;
    this.client = null;
    this.transport = null;
    this.currentRoomUrl = null;
    this.currentRoomName = null;

    this.stopHeartbeat();
    this.clearBotResponseTimeout();
    this.clearInitialGreetingTimeout();
    this.hasReceivedInitialGreeting = false;

    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }

    const audioElement = document.getElementById('pipecat-bot-audio');
    if (audioElement) {
      audioElement.remove();
    }
  }

  setMicrophoneEnabled(enabled: boolean): void {
    if (!this.client) return;

    this.isMicrophoneActive = enabled;
    try {
      const tracks = this.client.tracks();
      if (tracks?.local?.audio) {
        tracks.local.audio.enabled = enabled;
      }
    } catch (error) {
      console.error('Error controlling microphone:', error);
    }
  }

  getTransport(): PipecatTransport | null {
    return this.transport;
  }

  getIsConnected(): boolean {
    return this.isConnected;
  }

  getIsBotSpeaking(): boolean {
    return this.isBotSpeaking;
  }

  getIsUserSpeaking(): boolean {
    return this.isUserSpeaking;
  }

  async sendMessage(text: string): Promise<void> {
    if (!this.isConnected || !this.client) {
      throw new Error('Not connected');
    }
    console.log('Sending message:', text);
  }

  /**
   * Universal helper to send app messages to the bot.
   * Tries multiple methods to ensure compatibility across API versions.
   */
  private sendAppMessageToBot(messageType: string, data: any): boolean {
    if (!this.isConnected) {
      console.warn(`⚠️ Cannot send ${messageType} - not connected`);
      return false;
    }

    try {
      const client = this.client as any;
      const transport = this.transport as any;

      // Method 1: Try client.sendAppMessage (older API)
      if (client && typeof client.sendAppMessage === 'function') {
        client.sendAppMessage(messageType, data);
        console.log(`✅ ${messageType} sent via client.sendAppMessage`);
        return true;
      }

      // Method 2: Try transport.sendAppMessage
      if (transport && typeof transport.sendAppMessage === 'function') {
        transport.sendAppMessage(messageType, data);
        console.log(`✅ ${messageType} sent via transport.sendAppMessage`);
        return true;
      }

      // Method 3: Try transport.sendMessage with Daily format
      // Note: Need to ensure data is JSON serializable for postMessage
      if (transport && typeof transport.sendMessage === 'function') {
        // Deep clone to ensure serializable (removes functions, circular refs, etc.)
        const serializableData = JSON.parse(JSON.stringify(data));
        transport.sendMessage({
          label: messageType,
          type: messageType,
          data: serializableData
        });
        console.log(`✅ ${messageType} sent via transport.sendMessage`);
        return true;
      }

      // Method 4: Try accessing Daily call object directly
      if (transport && transport._daily && typeof transport._daily.sendAppMessage === 'function') {
        transport._daily.sendAppMessage({ type: messageType, data: data }, '*');
        console.log(`✅ ${messageType} sent via transport._daily.sendAppMessage`);
        return true;
      }

      // Method 5: Try dailyCallClient
      if (transport && transport.dailyCallClient && typeof transport.dailyCallClient.sendAppMessage === 'function') {
        transport.dailyCallClient.sendAppMessage({ type: messageType, data: data }, '*');
        console.log(`✅ ${messageType} sent via dailyCallClient.sendAppMessage`);
        return true;
      }

      console.warn(`⚠️ No method available to send ${messageType}`);
      return false;
    } catch (error) {
      console.error(`❌ Error sending ${messageType}:`, error);
      return false;
    }
  }

  /**
   * Send learning context to the bot via app message.
   * The bot waits for this before starting the conversation.
   */
  private sendLearningContext(learningContext: any): void {
    this.sendAppMessageToBot('learning-context', learningContext);
  }

  private sendUserTranscriptionToBackend(transcript: string): void {
    this.sendAppMessageToBot('user-transcription', {
      timestamp: Date.now(),
      transcription: transcript,
      type: 'user-transcription'
    });
  }

  signalPushToTalkStart(): void {
    if (!this.isConnected) return;

    this.isPushToTalkActive = true;
    this.pushToTalkStartTime = Date.now();

    this.sendAppMessageToBot('push-to-talk-start', { timestamp: this.pushToTalkStartTime });
  }

  signalUserStoppedSpeaking(): void {
    if (!this.isConnected) return;

    this.sendAppMessageToBot('user-stopped-speaking', { timestamp: Date.now() });
  }

  private waitForTranscription(timeoutMs: number = 5000): Promise<string> {
    return new Promise((resolve) => {
      if (this.finalUserTranscript) {
        resolve(this.finalUserTranscript);
        return;
      }

      this.isWaitingForTranscription = true;
      this.transcriptionResolve = resolve;

      this.transcriptionTimeout = setTimeout(() => {
        if (this.isWaitingForTranscription) {
          this.isWaitingForTranscription = false;
          this.transcriptionResolve = null;
          resolve(this.currentUserTranscript || '');
        }
      }, timeoutMs);
    });
  }

  async signalUserTurnComplete(): Promise<void> {
    if (!this.isConnected) return;

    const speechDurationMs = this.pushToTalkStartTime > 0
      ? Date.now() - this.pushToTalkStartTime
      : 0;

    this.isPushToTalkActive = false;
    this.pushToTalkStartTime = 0;

    try {
      const transcript = await this.waitForTranscription(5000);

      const sent = this.sendAppMessageToBot('user-turn-complete', {
        timestamp: Date.now(),
        transcript: transcript,
        hasTranscript: !!transcript,
        speechDurationMs: speechDurationMs
      });

      if (sent) {
        this.startBotResponseTimeout();
      }
    } catch (error) {
      console.error('Error sending turn complete:', error);
    }

    this.finalUserTranscript = '';
    this.currentUserTranscript = '';
  }
}

// Export singleton instance
export const pipecatLive = new PipecatLiveService();
