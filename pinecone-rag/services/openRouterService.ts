/**
 * OpenRouter Service for Vue
 * Now uses backend API to keep API keys secure
 * Backend handles AI-powered tutoring and roleplay with RAG integration
 */

export type Language = 'en' | 'ta' | 'te' | 'hi';

export interface Message {
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
  language?: string;
  audioUrl?: string;
  audioDuration?: number;  // Duration in seconds
}

interface OpenRouterServiceConfig {
  language: Language;
  learnerLevel?: string;
  moduleContext?: string;
  roleplayMode?: boolean;
  roleplayScenario?: string;
  roleplayCharacter?: string;
}

class OpenRouterService {
  private config: OpenRouterServiceConfig;
  private conversationHistory: Message[] = [];
  private sessionId: string;
  private apiBaseUrl: string;

  constructor(config: OpenRouterServiceConfig = { language: 'en' }) {
    this.config = config;
    const runtimeConfig = useRuntimeConfig();
    this.apiBaseUrl = runtimeConfig.public.API_BASE_URL as string;

    // Generate unique session ID for this service instance
    this.sessionId = `session_${Date.now()}_${Math.random().toString(36).substring(7)}`;

    // Initialize session on backend
    this.initializeSession();
  }

  /**
   * Initialize session on backend
   */
  private async initializeSession(): Promise<void> {
    try {
      const response = await fetch(`${this.apiBaseUrl}/api/openrouter/session/init`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          sessionId: this.sessionId,
          config: this.config
        })
      });

      if (!response.ok) {
        console.error('Failed to initialize session');
      }
    } catch (error) {
      console.error('Error initializing session:', error);
    }
  }


  /**
   * Send message via backend API
   */
  async sendMessage(message: string): Promise<string> {
    try {
      const response = await fetch(`${this.apiBaseUrl}/api/openrouter/message`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          sessionId: this.sessionId,
          message
        })
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || errorData.message || `API error: ${response.status}`);
      }

      const data = await response.json();
      const aiResponse = data.response || '';

      if (!aiResponse) {
        throw new Error('No response from API');
      }

      // Update local conversation history
      this.conversationHistory.push({
        role: 'user',
        content: message,
        timestamp: new Date().toISOString()
      });

      this.conversationHistory.push({
        role: 'assistant',
        content: aiResponse,
        timestamp: new Date().toISOString()
      });

      return aiResponse;
    } catch (error: any) {
      console.error('Error sending message:', error);
      throw error;
    }
  }

  /**
   * Send message with streaming response
   * Note: OpenRouter supports streaming but for simplicity we use non-streaming
   * Can be enhanced later with SSE streaming
   */
  async sendMessageStream(
    message: string,
    onChunk: (text: string) => void
  ): Promise<string> {
    // For now, use regular sendMessage and simulate streaming
    const response = await this.sendMessage(message);

    // Simulate streaming by sending chunks
    const words = response.split(' ');
    let chunkText = '';
    for (const word of words) {
      chunkText += word + ' ';
      onChunk(chunkText);
      await new Promise((resolve) => setTimeout(resolve, 30));
    }

    return response;
  }

  /**
   * Update service configuration
   */
  async updateConfig(config: Partial<OpenRouterServiceConfig>): Promise<void> {
    const previousLanguage = this.config.language;
    this.config = { ...this.config, ...config };

    // Update config on backend
    try {
      await fetch(`${this.apiBaseUrl}/api/openrouter/session/config`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          sessionId: this.sessionId,
          config
        })
      });

      // Reset local conversation when language changes
      if (config.language && config.language !== previousLanguage) {
        // console.log(`Language changed from ${previousLanguage} to ${config.language}`);
        this.conversationHistory = [];
      }
    } catch (error) {
      console.error('Error updating config:', error);
    }
  }

  /**
   * Reset conversation history
   */
  async resetChat(): Promise<void> {
    try {
      await fetch(`${this.apiBaseUrl}/api/openrouter/session/reset`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          sessionId: this.sessionId
        })
      });

      this.conversationHistory = [];
    } catch (error) {
      console.error('Error resetting chat:', error);
    }
  }

  /**
   * Get current configuration
   */
  getConfig(): OpenRouterServiceConfig {
    return { ...this.config };
  }

  /**
   * Get conversation history
   */
  getHistory(): Message[] {
    return [...this.conversationHistory];
  }
}

export default OpenRouterService;
