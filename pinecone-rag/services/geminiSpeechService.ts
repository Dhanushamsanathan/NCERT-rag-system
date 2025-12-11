/**
 * Gemini Speech Service
 * Handles text-to-speech using backend Gemini TTS API
 * Backend keeps API keys secure
 */

import type { Language } from './geminiService';

export class GeminiSpeechService {
  private currentSource: AudioBufferSourceNode | null = null;
  private audioContext: AudioContext | null = null; // Reuse single context
  private gainNode: GainNode | null = null; // Gain node for mute/unmute
  private apiBaseUrl: string;
  private audioQueue: Uint8Array[] = []; // Queue for streaming audio chunks
  private isPlaying: boolean = false;
  private streamingEnabled: boolean = true; // Use streaming by default
  private scheduledSources: AudioBufferSourceNode[] = []; // Track scheduled audio sources
  private nextPlayTime: number = 0; // Track next scheduled playback time
  private isMuted: boolean = false; // Track mute state

  constructor() {
    const runtimeConfig = useRuntimeConfig();
    this.apiBaseUrl = runtimeConfig.public.API_BASE_URL as string;
  }

  isSupported(): boolean {
    return typeof AudioContext !== 'undefined' && typeof fetch !== 'undefined';
  }

  async speak(
    text: string,
    language: Language = 'en',
    onEnd?: () => void,
    onError?: (error: string) => void
  ): Promise<void> {
    if (!this.isSupported()) {
      if (onError) {
        onError('Audio playback is not supported in your browser');
      }
      return;
    }

    // Use streaming if enabled
    if (this.streamingEnabled) {
      await this.speakStreaming(text, language, onEnd, onError);
    } else {
      await this.speakNonStreaming(text, language, onEnd, onError);
    }
  }

  /**
   * Streaming TTS - lower latency, progressive audio playback
   * Plays audio chunks as they arrive for minimal latency
   */
  private async speakStreaming(
    text: string,
    language: Language,
    onEnd?: () => void,
    onError?: (error: string) => void
  ): Promise<void> {
    try {
      console.log(`Generating speech for ${language} via streaming backend:`, text);

      // Stop any currently playing audio
      this.stop();

      // Initialize audio context
      if (!this.audioContext || this.audioContext.state === 'closed') {
        this.audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
        // Create gain node for volume control (mute/unmute)
        this.gainNode = this.audioContext.createGain();
        this.gainNode.connect(this.audioContext.destination);
        this.gainNode.gain.value = 1; // Start unmuted
        this.isMuted = false;
      } else if (this.audioContext.state === 'suspended') {
        await this.audioContext.resume();
      }

      const audioContext = this.audioContext;

      // Initialize playback scheduling
      this.nextPlayTime = audioContext.currentTime;
      this.scheduledSources = [];
      this.isPlaying = true;

      // Call backend streaming TTS API
      const response = await fetch(`${this.apiBaseUrl}/api/tts/synthesize-stream`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          text,
          language
        })
      });

      if (!response.ok) {
        throw new Error(`TTS streaming API error: ${response.status}`);
      }

      const reader = response.body?.getReader();
      if (!reader) {
        throw new Error('No response body reader available');
      }

      const decoder = new TextDecoder();
      let buffer = '';
      let chunkCount = 0;
      let totalChunks = 0;

      // Read and play streaming response progressively
      while (true) {
        const { done, value } = await reader.read();

        if (done) {
          break;
        }

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = JSON.parse(line.substring(6));

            if (data.type === 'chunk' && data.data) {
              chunkCount++;

              // Convert base64 to binary
              const audioBytes = Uint8Array.from(atob(data.data), (c) => c.charCodeAt(0));

              console.log(`Chunk ${data.chunkNumber} received (${audioBytes.length} bytes), playing immediately`);

              // Play this chunk immediately (queue it sequentially)
              // We'll attach onEnd callback to the last chunk after we know the total
              await this.playChunkProgressive(audioBytes, audioContext);

            } else if (data.type === 'done') {
              totalChunks = data.totalChunks;
              console.log(`Streaming complete, total chunks: ${totalChunks}, time-to-first-audio: ${data.timeToFirstAudioMs}ms`);

              // Attach onEnd callback to the last scheduled source
              if (this.scheduledSources.length > 0) {
                const lastSource = this.scheduledSources[this.scheduledSources.length - 1];
                lastSource.onended = () => {
                  console.log('Last audio chunk finished playing');
                  this.isPlaying = false;
                  this.scheduledSources = [];
                  if (onEnd) {
                    onEnd();
                  }
                };
              } else {
                // No chunks scheduled, call onEnd immediately
                this.isPlaying = false;
                if (onEnd) {
                  onEnd();
                }
              }
            } else if (data.type === 'error') {
              throw new Error(data.message || 'Streaming error');
            }
          }
        }
      }

      if (chunkCount === 0) {
        throw new Error('No audio chunks received');
      }

      console.log(`Progressive playback complete, queued ${chunkCount} chunks`);

    } catch (error) {
      console.error('Streaming TTS error:', error);
      this.isPlaying = false;
      if (onError) {
        onError(
          `Failed to generate speech: ${error instanceof Error ? error.message : 'Unknown error'}`
        );
      }
    }
  }

  /**
   * Play audio chunk progressively as it arrives
   * Schedules each chunk to play seamlessly after the previous one
   */
  private async playChunkProgressive(
    audioBytes: Uint8Array,
    audioContext: AudioContext
  ): Promise<void> {
    try {
      const sampleRate = 24000;
      const numChannels = 1;

      // Convert Uint8Array to Int16Array (16-bit PCM)
      const int16Array = new Int16Array(audioBytes.buffer);
      const numSamples = int16Array.length;

      // Create an AudioBuffer for this chunk
      const audioBuffer = audioContext.createBuffer(numChannels, numSamples, sampleRate);

      // Convert Int16 to Float32 (AudioBuffer expects Float32)
      const channelData = audioBuffer.getChannelData(0);
      for (let i = 0; i < numSamples; i++) {
        channelData[i] = int16Array[i] / 32768.0;
      }

      // Create source for this chunk
      const source = audioContext.createBufferSource();
      source.buffer = audioBuffer;
      // Connect through gain node for mute/unmute control
      if (this.gainNode) {
        source.connect(this.gainNode);
      } else {
        source.connect(audioContext.destination);
      }

      // Schedule this chunk to play at the next available time
      const playTime = Math.max(this.nextPlayTime, audioContext.currentTime);
      source.start(playTime);

      // Update next play time for seamless playback
      this.nextPlayTime = playTime + audioBuffer.duration;

      // Track this source
      this.scheduledSources.push(source);

      console.log(
        `Scheduled chunk: ${numSamples} samples, duration: ${audioBuffer.duration.toFixed(3)}s, playing at: ${playTime.toFixed(3)}s`
      );
    } catch (error) {
      console.error('Error playing audio chunk:', error);
    }
  }

  /**
   * Non-streaming TTS - fallback method
   */
  private async speakNonStreaming(
    text: string,
    language: Language,
    onEnd?: () => void,
    onError?: (error: string) => void
  ): Promise<void> {
    try {
      console.log(`Generating speech for ${language} via backend:`, text);

      // Call backend TTS API
      const response = await fetch(`${this.apiBaseUrl}/api/tts/synthesize`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          text,
          language
        })
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.message || `TTS API error: ${response.status}`);
      }

      const data = await response.json();

      if (!data.success || !data.audioData) {
        throw new Error('No audio data received from backend');
      }

      console.log('Successfully received audio data, length:', data.audioData.length);

      // Convert base64 to binary
      const audioBytes = Uint8Array.from(atob(data.audioData), (c) => c.charCodeAt(0));

      console.log('Audio generation complete, starting playback');

      // Play the audio
      await this.playAudio(audioBytes, onEnd, onError);
    } catch (error) {
      console.error('TTS error:', error);
      if (onError) {
        onError(
          `Failed to generate speech: ${error instanceof Error ? error.message : 'Unknown error'}`
        );
      }
    }
  }

  private async playAudio(
    audioBytes: Uint8Array,
    onEnd?: () => void,
    onError?: (error: string) => void
  ): Promise<void> {
    try {
      // Stop any currently playing audio
      this.stop();

      // Create or reuse audio context
      if (!this.audioContext || this.audioContext.state === 'closed') {
        this.audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
        // Create gain node for volume control (mute/unmute)
        this.gainNode = this.audioContext.createGain();
        this.gainNode.connect(this.audioContext.destination);
        this.gainNode.gain.value = 1; // Start unmuted
        this.isMuted = false;
        console.log('Created new AudioContext for Gemini TTS');
      } else if (this.audioContext.state === 'suspended') {
        await this.audioContext.resume();
        console.log('Resumed existing AudioContext');
      }

      const audioContext = this.audioContext;

      // Gemini TTS returns raw PCM data at 24kHz, 16-bit, mono
      // We need to create an AudioBuffer directly from the PCM data
      const sampleRate = 24000;
      const numChannels = 1; // mono

      // Convert Uint8Array to Int16Array (16-bit PCM)
      const int16Array = new Int16Array(audioBytes.buffer);
      const numSamples = int16Array.length;

      // Create an AudioBuffer
      const audioBuffer = audioContext.createBuffer(numChannels, numSamples, sampleRate);

      // Get the channel data and convert Int16 to Float32 (AudioBuffer expects Float32)
      const channelData = audioBuffer.getChannelData(0);
      for (let i = 0; i < numSamples; i++) {
        // Convert from Int16 (-32768 to 32767) to Float32 (-1.0 to 1.0)
        channelData[i] = int16Array[i] / 32768.0;
      }

      console.log(
        `Created audio buffer: ${numSamples} samples at ${sampleRate}Hz, duration: ${audioBuffer.duration.toFixed(2)}s`
      );

      // Create source
      const source = audioContext.createBufferSource();
      source.buffer = audioBuffer;
      this.currentSource = source;

      // Connect to output through gain node for mute/unmute control
      if (this.gainNode) {
        source.connect(this.gainNode);
      } else {
        source.connect(audioContext.destination);
      }

      // Handle playback end
      source.onended = () => {
        this.currentSource = null;
        if (onEnd) onEnd();
      };

      // Start playback
      source.start();
    } catch (error) {
      console.error('Audio playback error:', error);
      this.currentSource = null;
      if (onError) {
        onError(
          `Failed to play audio: ${error instanceof Error ? error.message : 'Unknown error'}`
        );
      }
    }
  }

  stop(): void {
    console.log('Stop called - stopping all audio playback');

    // Stop current single source (for non-streaming)
    if (this.currentSource) {
      try {
        this.currentSource.stop(0); // Stop immediately
        this.currentSource.disconnect();
        console.log('Gemini TTS playback stopped');
      } catch (error) {
        // Source might already be stopped, ignore error
        console.log('Audio source already stopped or disconnected');
      }
      this.currentSource = null;
    }

    // Stop all scheduled sources (for streaming)
    if (this.scheduledSources.length > 0) {
      console.log(`Stopping ${this.scheduledSources.length} scheduled audio chunks`);
      for (const source of this.scheduledSources) {
        try {
          source.stop(0); // Stop immediately
          source.disconnect();
        } catch (error) {
          // Source might already be stopped or not started yet, ignore error
          console.log('Source already stopped:', error);
        }
      }
      this.scheduledSources = [];
      console.log('All scheduled audio chunks stopped');
    }

    this.isPlaying = false;
    this.nextPlayTime = 0;

    // Keep the AudioContext alive and running
    // This single context is reused for all Gemini TTS playback
  }

  isLanguageSupported(language: Language): boolean {
    // Backend supports English, Tamil, Telugu, and Hindi
    const supportedLanguages: Language[] = ['en', 'ta', 'te', 'hi'];
    return this.isSupported() && supportedLanguages.includes(language);
  }

  // Check if audio is currently playing
  isSpeaking(): boolean {
    return this.isPlaying || this.currentSource !== null || this.scheduledSources.length > 0;
  }

  /**
   * Mute audio without stopping playback
   */
  mute(): void {
    if (this.gainNode) {
      this.gainNode.gain.value = 0;
      this.isMuted = true;
      console.log('Audio muted');
    }
  }

  /**
   * Unmute audio
   */
  unmute(): void {
    if (this.gainNode) {
      this.gainNode.gain.value = 1;
      this.isMuted = false;
      console.log('Audio unmuted');
    }
  }

  /**
   * Check if audio is currently muted
   */
  isMutedAudio(): boolean {
    return this.isMuted;
  }
}

// Export singleton instance
export const geminiSpeech = new GeminiSpeechService();
