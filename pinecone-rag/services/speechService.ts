/**
 * Speech Service
 * Handles speech recognition and text-to-speech
 * Uses browser Web Speech API for recognition
 * Uses Gemini TTS (via backend) for speech synthesis
 */

import type { Language } from './geminiService';
import { geminiSpeech } from './geminiSpeechService';

// Language codes for Web Speech API (used for speech recognition)
const LANGUAGE_CODES: Record<Language, string> = {
  en: 'en-US',
  ta: 'ta-IN',
  te: 'te-IN',
  hi: 'hi-IN'
};

export class SpeechRecognitionService {
  private recognition: any = null;
  private language: Language = 'en';
  private isListening: boolean = false;

  constructor() {
    if (!this.isSupported()) {
      console.warn('Speech recognition is not supported in this browser');
    }
  }

  isSupported(): boolean {
    return 'webkitSpeechRecognition' in window || 'SpeechRecognition' in window;
  }

  initialize(language: Language = 'en'): void {
    if (!this.isSupported()) {
      throw new Error('Speech recognition is not supported in this browser');
    }

    const SpeechRecognition =
      (window as any).webkitSpeechRecognition || (window as any).SpeechRecognition;
    this.recognition = new SpeechRecognition();
    this.language = language;

    this.recognition.continuous = false;
    this.recognition.interimResults = true;
    this.recognition.lang = LANGUAGE_CODES[language];
  }

  startListening(
    onResult: (transcript: string, isFinal: boolean) => void,
    onError?: (error: string) => void,
    onEnd?: () => void
  ): void {
    if (!this.recognition) {
      this.initialize(this.language);
    }

    this.recognition.onresult = (event: any) => {
      const results = event.results;
      const lastResult = results[results.length - 1];
      const transcript = lastResult[0].transcript;
      const isFinal = lastResult.isFinal;

      onResult(transcript, isFinal);
    };

    this.recognition.onerror = (event: any) => {
      console.error('Speech recognition error:', event.error);
      if (onError) {
        onError(event.error);
      }
      this.isListening = false;
    };

    this.recognition.onend = () => {
      this.isListening = false;
      if (onEnd) {
        onEnd();
      }
    };

    try {
      this.recognition.start();
      this.isListening = true;
    } catch (error) {
      console.error('Error starting speech recognition:', error);
      if (onError) {
        onError('Failed to start speech recognition');
      }
    }
  }

  stopListening(): void {
    if (this.recognition && this.isListening) {
      this.recognition.stop();
      this.isListening = false;
    }
  }

  setLanguage(language: Language): void {
    this.language = language;
    if (this.recognition) {
      this.recognition.lang = LANGUAGE_CODES[language];
    }
  }

  getIsListening(): boolean {
    return this.isListening;
  }
}

export class TextToSpeechService {
  private language: Language = 'en';

  constructor() {
    // Using Gemini TTS via backend API
  }

  isSupported(): boolean {
    // Check if Gemini TTS is supported
    return geminiSpeech.isSupported();
  }

  isLanguageSupported(language: Language): boolean {
    // All languages supported via Gemini TTS backend
    return geminiSpeech.isLanguageSupported(language);
  }

  speak(
    text: string,
    language?: Language,
    onEnd?: () => void,
    onError?: (error: string) => void
  ): void {
    const targetLanguage = language || this.language;
    console.log(`TextToSpeechService.speak called for language: ${targetLanguage} via Gemini TTS`);

    // Use Gemini TTS for all languages
    if (!geminiSpeech.isSupported()) {
      if (onError) {
        onError('Text-to-speech is not supported');
      }
      return;
    }

    // Mute any currently playing speech (unmute happens when it finishes naturally)
    if (geminiSpeech.isSpeaking()) {
      geminiSpeech.mute();
    }

    // Use Gemini TTS (backend API with native audio support for all languages)
    geminiSpeech.speak(
      text,
      targetLanguage,
      () => {
        // Unmute when speech ends naturally
        geminiSpeech.unmute();
        if (onEnd) onEnd();
      },
      (error) => {
        // Unmute on error as well
        geminiSpeech.unmute();
        if (onError) onError(error);
      }
    );
  }

  stop(): void {
    // Mute Gemini speech instead of stopping (stop doesn't work reliably)
    geminiSpeech.mute();
  }

  pause(): void {
    // Mute instead of pause
    geminiSpeech.mute();
  }

  resume(): void {
    // Unmute to resume
    geminiSpeech.unmute();
  }

  mute(): void {
    // Mute audio without stopping playback
    geminiSpeech.mute();
  }

  unmute(): void {
    // Unmute audio
    geminiSpeech.unmute();
  }

  setLanguage(language: Language): void {
    this.language = language;
  }

  isSpeaking(): boolean {
    return geminiSpeech.isSpeaking();
  }

  isMuted(): boolean {
    return geminiSpeech.isMutedAudio();
  }
}

// Export singleton instances
export const speechRecognition = new SpeechRecognitionService();
export const textToSpeech = new TextToSpeechService();
