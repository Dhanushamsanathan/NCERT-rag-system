/**
 * Audio Recording Service
 * Uses RecordRTC to record audio as WAV for OpenRouter transcription
 * WAV is lightweight, widely supported, and OpenRouter compatible
 */

import RecordRTC from 'recordrtc';

export type Language = 'en' | 'ta' | 'te' | 'hi';

interface AudioRecordingConfig {
  language: Language;
  onTranscript: (text: string, isFinal: boolean) => void;
  onError: (error: string) => void;
  onStop: () => void;
}

class AudioRecordingService {
  private recorder: RecordRTC | null = null;
  private config: AudioRecordingConfig | null = null;
  private apiBaseUrl: string;
  private isRecording: boolean = false;
  private stream: MediaStream | null = null;

  constructor() {
    const runtimeConfig = useRuntimeConfig();
    this.apiBaseUrl = runtimeConfig.public.API_BASE_URL as string;
  }

  /**
   * Check if audio recording is supported
   */
  isSupported(): boolean {
    return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
  }

  /**
   * Start recording audio as WAV
   */
  async startRecording(config: AudioRecordingConfig): Promise<void> {
    if (!this.isSupported()) {
      config.onError('Audio recording is not supported in your browser');
      return;
    }

    if (this.isRecording) {
      console.warn('Already recording');
      return;
    }

    try {
      this.config = config;

      // Request microphone access
      this.stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          sampleRate: 16000,
          echoCancellation: true,
          noiseSuppression: true
        }
      });

      // Create RecordRTC instance for WAV recording
      this.recorder = new RecordRTC(this.stream, {
        type: 'audio',
        mimeType: 'audio/wav',
        recorderType: RecordRTC.StereoAudioRecorder,
        numberOfAudioChannels: 1,
        desiredSampRate: 16000,
        timeSlice: 1000,
        ondataavailable: async (blob: Blob) => {
          // This is called periodically, but we'll process on stop
        }
      });

      // Start recording
      this.recorder.startRecording();
      this.isRecording = true;

      console.log('WAV recording started with RecordRTC');
    } catch (error: any) {
      console.error('Error starting recording:', error);
      config.onError('Failed to start recording: ' + (error.message || 'Unknown error'));
      this.isRecording = false;

      // Clean up stream if it was created
      if (this.stream) {
        this.stream.getTracks().forEach(track => track.stop());
        this.stream = null;
      }
    }
  }

  /**
   * Stop recording audio and process it
   */
  stopRecording(): void {
    if (this.recorder && this.isRecording) {
      this.isRecording = false;

      this.recorder.stopRecording(async () => {
        // Get the recorded blob
        const audioBlob = this.recorder!.getBlob();

        // Stop all tracks in the stream
        if (this.stream) {
          this.stream.getTracks().forEach(track => track.stop());
          this.stream = null;
        }

        // Process the recording
        await this.processRecording(audioBlob);

        // Clean up recorder
        this.recorder = null;
      });
    }
  }

  /**
   * Process recorded audio and send to backend for transcription
   */
  private async processRecording(audioBlob: Blob): Promise<void> {
    if (!this.config) {
      console.warn('No config available for processing');
      return;
    }

    try {
      console.log('Audio blob created:', {
        size: audioBlob.size,
        type: audioBlob.type
      });

      // Convert blob to base64
      const base64Audio = await this.blobToBase64(audioBlob);

      // Send to backend for transcription
      const response = await fetch(`${this.apiBaseUrl}/api/audio/transcribe`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          audioBase64: base64Audio,
          mimeType: 'audio/wav',
          language: this.config.language
        })
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.message || `Transcription failed: ${response.status}`);
      }

      const data = await response.json();

      if (data.success && data.transcription) {
        // Call callback with transcription
        this.config.onTranscript(data.transcription, true);
      } else {
        throw new Error('No transcription received');
      }
    } catch (error: any) {
      console.error('Error processing recording:', error);
      if (this.config) {
        this.config.onError('Transcription failed: ' + (error.message || 'Unknown error'));
      }
    } finally {
      // Call stop callback
      if (this.config) {
        this.config.onStop();
      }
    }
  }

  /**
   * Convert blob to base64
   */
  private blobToBase64(blob: Blob): Promise<string> {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onloadend = () => {
        const base64String = reader.result as string;
        // Remove data URL prefix (e.g., "data:audio/wav;base64,")
        const base64Data = base64String.split(',')[1];
        resolve(base64Data);
      };
      reader.onerror = reject;
      reader.readAsDataURL(blob);
    });
  }

  /**
   * Check if currently recording
   */
  isCurrentlyRecording(): boolean {
    return this.isRecording;
  }
}

// Export singleton instance
export const audioRecordingService = new AudioRecordingService();

// Export class for testing
export default AudioRecordingService;
