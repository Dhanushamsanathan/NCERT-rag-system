/**
 * Gemini Service for Vue
 * Handles AI-powered tutoring with RAG integration
 * Based on the original React implementation
 */

import { GoogleGenerativeAI } from '@google/generative-ai';

export type Language = 'en' | 'ta' | 'te' | 'hi';

export interface Message {
  role: 'user' | 'ai';
  content: string;
  timestamp: string;
}

interface GeminiServiceConfig {
  language: Language;
  learnerLevel?: string;
  moduleContext?: string;
}

interface VectorSearchResult {
  chunk_id: string;
  study_material_id: string;
  chunk_index: number;
  content: string;
  similarity_score: number;
  metadata: Record<string, any>;
  filename: string;
}

interface VectorSearchResponse {
  results: VectorSearchResult[];
  total_results: number;
  query: string;
}

class GeminiService {
  private model: any;
  private chat: any = null;
  private config: GeminiServiceConfig;
  private isInitialized: boolean = false;
  private documentsApiUrl: string;
  private apiKey: string | undefined;
  private languageChanged: boolean = false;

  constructor(config: GeminiServiceConfig = { language: 'en' }) {
    this.config = config;
    const runtimeConfig = useRuntimeConfig();
    this.apiKey = runtimeConfig.public.GEMINI_API_KEY as string;
    this.documentsApiUrl = runtimeConfig.public.DOCUMENTS_API_URL as string;

    if (!this.apiKey) {
      console.warn('Gemini API key not provided. Running in mock mode.');
      this.isInitialized = false;
      return;
    }

    try {
      const genAI = new GoogleGenerativeAI(this.apiKey);
      // Using the same model as React implementation
      this.model = genAI.getGenerativeModel({ model: 'gemini-2.5-flash-lite' });
      this.isInitialized = true;
    } catch (error) {
      console.error('Failed to initialize Gemini AI:', error);
      this.isInitialized = false;
    }
  }

  private getSystemPrompt(): string {
    const languageNote: Record<Language, string> = {
      en: 'Match the user\'s language naturally in English.',
      ta: 'Match the user\'s language — Tamil, English, or Tanglish mix.',
      te: 'Match the user\'s language — Telugu, English, or Tenglish mix.',
      hi: 'Match the user\'s language — Hindi, English, or Hinglish mix.'
    };

    return `Role: You are a human tutor who uses only the provided study materials.

Tone & Style: Speak naturally like a patient teacher. Keep language plain, friendly, and concise. ${languageNote[this.config.language]}

Structure (always follow):
1. Summary: Begin with a single-sentence summary of the answer.
2. Steps: Provide a numbered, step-by-step explanation. Each step should be 1–2 short sentences.
3. Example: Give one brief, practical example or analogy.
4. Next Step: Finish with one clear, actionable next step or a learner question.

Formatting rules:
• Use short paragraphs
• Use numbered lists for steps (3–6 steps maximum)
• Keep answers focused and concise
• Bold important terms

Strict constraints:
• Use only content from the provided study materials.
• Never mention AI, training, data sources, file names, page numbers, or citations.
• If a topic is not covered, reply only: "I cannot help with that topic as it is not covered in the study materials." (Say nothing else.)
• Match the language of the user's query — respond in the same language or language mix.

${this.config.moduleContext ? `Current Topic: ${this.config.moduleContext}` : ''}
${this.config.learnerLevel ? `Learner Level: ${this.config.learnerLevel} - Adjust complexity accordingly` : ''}

Goal: Always teach clearly and help the learner move forward.`;
  }

  private getMockResponse(message: string): string {
    const responses: Record<Language, string[]> = {
      en: [
        'Welcome to Module 1: Foundation Concepts! I am excited to help you learn about Introduction to Sales. What is your current experience with sales?',
        'Great to see you starting with Introduction to Sales - Foundation Concepts! What specific aspect of sales fundamentals would you like to explore first?',
        'Hello! I am your AI mentor for Module 1: Introduction to Sales. What brought you to this course, and what are you hoping to achieve?',
        'Excellent choice starting with Foundation Concepts! Understanding sales fundamentals is crucial. What is your background in customer interactions?'
      ],
      ta: [
        'தொகுதி 1: அடிப்படைக் கருத்துகளுக்கு வரவேற்கிறோம்! விற்பனை அறிமுகத்தைப் பற்றி கற்றுக்கொள்ள உங்களுக்கு உதவ நான் ஆர்வமாக உள்ளேன். விற்பனையில் உங்களுடைய தற்போதைய அனுபவம் என்ன?',
        'விற்பனை அறிமுகம் - அடிப்படைக் கருத்துகளுடன் தொடங்குவதை பார்த்து மகிழ்ச்சியடைகிறேன்! விற்பனை அடிப்படைகளின் குறிப்பிட்ட அம்சத்தை நீங்கள் முதலில் ஆராய விரும்புகிறீர்களா?'
      ],
      te: [
        'మాడ్యూల్ 1: ఫండమెంటల్ కాన్సెప్ట్లకు స్వాగతం! అమ్మకాల పరిచయం గురించి నేర్చుకోవడానికి మీకు సహాయం చేయడానికి నేను ఉత్సాహంగా ఉన్నాను. అమ్మకాలలో మీ ప్రస్తుత అనుభవం ఏమిటి?',
        'అమ్మకాల పరిచయం - ఫండమెంటల్ కాన్సెప్ట్లతో ప్రారంభించడం చూడటం గొప్పగా ఉంది! మీరు మొదట అన్వేషించాలనుకునే అమ్మకాల ప్రాథమిక అంశాలు ఏమిటి?'
      ],
      hi: [
        'मॉड्यूल 1: फंडामेंटल कॉन्सेप्ट्स में आपका स्वागत है! मैं सेल्स का परिचय सिखाने में आपकी मदद करने के लिए उत्साहित हूं। आपका बिक्री में वर्तमान अनुभव क्या है?',
        'बिक्री परिचय - फंडामेंटल कॉन्सेप्ट्स के साथ शुरू करते देखकर मुझे खुशी है! आप बिक्री के मूल सिद्धांतों के किस विशिष्ट पहलू को पहले एक्सप्लोर करना चाहेंगे?'
      ]
    };

    const langResponses = responses[this.config.language] || responses.en;

    // Return a contextual response based on the message
    if (
      message.toLowerCase().includes('greeting') ||
      message.toLowerCase().includes('hello') ||
      message.toLowerCase().includes('start')
    ) {
      return langResponses[2] || langResponses[0];
    }
    return langResponses[Math.floor(Math.random() * langResponses.length)];
  }

  async initializeChat(conversationHistory: Message[] = []): Promise<void> {
    const history = conversationHistory.map((msg) => ({
      role: msg.role === 'ai' ? 'model' : 'user',
      parts: [{ text: msg.content }]
    }));

    this.chat = this.model.startChat({
      history,
      generationConfig: {
        temperature: 0.3, // Slightly increased for more natural, educational responses
        topK: 40,
        topP: 0.95,
        maxOutputTokens: 2048 // Increased to allow for detailed point-by-point explanations
      }
    });
  }

  private async retrieveStudyMaterials(query: string): Promise<string> {
    try {
      const response = await fetch(`${this.documentsApiUrl}/api/documents/search`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          query,
          matchCount: 5 // Increased from 3 to 5 for more comprehensive context
        })
      });

      if (!response.ok) {
        console.warn('Failed to retrieve study materials:', response.statusText);
        return '';
      }

      const searchResults: VectorSearchResponse = await response.json();

      if (searchResults.results.length === 0) {
        return '';
      }

      // Format study materials WITHOUT any citations, filenames, or metadata
      // Just provide the clean content for the tutor to use
      const contextText = searchResults.results
        .map((result) => result.content)
        .join('\n\n---\n\n');

      return contextText;
    } catch (error) {
      console.warn('Failed to retrieve study materials:', error);
      return '';
    }
  }

  async sendMessage(message: string): Promise<string> {
    // Return mock response if API is not initialized
    if (!this.isInitialized) {
      console.log('Using mock response (no API key)');
      return this.getMockResponse(message);
    }

    if (!this.chat) {
      await this.initializeChat();
    }

    try {
      // Retrieve relevant study materials
      const studyContext = await this.retrieveStudyMaterials(message);

      // Build contextualized message
      let contextualizedMessage: string;

      // Check if we need to include system prompt (first message or language changed)
      const shouldIncludeSystemPrompt = this.chat.getHistory().length === 0 || this.languageChanged;

      if (shouldIncludeSystemPrompt) {
        // Reset the language changed flag
        this.languageChanged = false;

        // First message or language changed - include system prompt and context
        if (studyContext) {
          contextualizedMessage = `${this.getSystemPrompt()}\n\n=== RELEVANT INFORMATION FROM YOUR STUDY MATERIALS ===\n${studyContext}\n\n=== END OF INFORMATION ===\n\nREMEMBER:
- This information is your ONLY knowledge source
- Do NOT mention where this information came from
- Structure your response clearly (bullet points, numbered lists when helpful)
- Explain naturally, as a teacher would
- DO NOT cite sources, page numbers, or document names\n\nLearner's question: ${message}\n\nTeach them clearly and naturally:`;
        } else {
          contextualizedMessage = `${this.getSystemPrompt()}\n\nNo relevant information found in your study materials for this topic. You must respond EXACTLY: "I cannot help with that topic as it is not covered in the study materials."\n\nLearner asked: ${message}`;
        }
      } else {
        // Subsequent message - include context if available
        if (studyContext) {
          contextualizedMessage = `=== RELEVANT INFORMATION ===\n${studyContext}\n\n=== END OF INFORMATION ===\n\nREMEMBER:
- Use ONLY this information
- No citations or source mentions
- Clear, natural teacher-like response\n\nQuestion: ${message}\n\nTeach naturally:`;
        } else {
          contextualizedMessage = `No information found. Respond EXACTLY: "I cannot help with that topic as it is not covered in the study materials."\n\nQuestion: ${message}`;
        }
      }

      // Send with retry logic for rate limits
      const result = await this.sendWithRetry(async () => {
        return await this.chat.sendMessage(contextualizedMessage);
      });

      const response = await result.response;
      return response.text();
    } catch (error: any) {
      console.error('Error sending message to Gemini:', error);

      // Check if it is a rate limit error
      if (error?.message?.includes('429') || error?.message?.includes('quota') || error?.message?.includes('rate limit')) {
        throw new Error('Rate limit exceeded. Please wait a moment and try again, or consider using a different Gemini API model.');
      }

      // Fall back to mock response on other API errors
      console.log('Falling back to mock response due to API error');
      return this.getMockResponse(message);
    }
  }

  /**
   * Retry helper with exponential backoff
   */
  private async sendWithRetry<T>(
    fn: () => Promise<T>,
    maxRetries: number = 3,
    baseDelay: number = 1000
  ): Promise<T> {
    for (let attempt = 0; attempt < maxRetries; attempt++) {
      try {
        return await fn();
      } catch (error: any) {
        const isRateLimitError =
          error?.message?.includes('429') ||
          error?.message?.includes('quota') ||
          error?.message?.includes('rate limit') ||
          error?.status === 429;

        // Do not retry if it is not a rate limit error, or if we are on the last attempt
        if (!isRateLimitError || attempt === maxRetries - 1) {
          throw error;
        }

        // Calculate delay with exponential backoff
        const delay = baseDelay * Math.pow(2, attempt);
        console.log(`Rate limit hit. Retrying in ${delay}ms... (attempt ${attempt + 1}/${maxRetries})`);

        await new Promise(resolve => setTimeout(resolve, delay));
      }
    }

    throw new Error('Max retries exceeded');
  }

  async sendMessageStream(
    message: string,
    onChunk: (text: string) => void
  ): Promise<string> {
    // Return mock response if API is not initialized
    if (!this.isInitialized) {
      console.log('Using mock streaming response (no API key)');
      const mockResponse = this.getMockResponse(message);
      // Simulate streaming by sending chunks
      const words = mockResponse.split(' ');
      let chunkText = '';
      for (const word of words) {
        chunkText += word + ' ';
        onChunk(chunkText);
        await new Promise((resolve) => setTimeout(resolve, 50)); // Small delay to simulate streaming
      }
      return mockResponse;
    }

    if (!this.chat) {
      await this.initializeChat();
    }

    try {
      // Retrieve relevant study materials
      const studyContext = await this.retrieveStudyMaterials(message);

      // Build contextualized message
      let contextualizedMessage: string;

      // Check if we need to include system prompt (first message or language changed)
      const shouldIncludeSystemPrompt = this.chat.getHistory().length === 0 || this.languageChanged;

      if (shouldIncludeSystemPrompt) {
        // Reset the language changed flag
        this.languageChanged = false;

        // First message or language changed - include system prompt and context
        if (studyContext) {
          contextualizedMessage = `${this.getSystemPrompt()}\n\n=== RELEVANT INFORMATION FROM YOUR STUDY MATERIALS ===\n${studyContext}\n\n=== END OF INFORMATION ===\n\nREMEMBER:
- This information is your ONLY knowledge source
- Do NOT mention where this information came from
- Structure your response clearly (bullet points, numbered lists when helpful)
- Explain naturally, as a teacher would
- DO NOT cite sources, page numbers, or document names\n\nLearner's question: ${message}\n\nTeach them clearly and naturally:`;
        } else {
          contextualizedMessage = `${this.getSystemPrompt()}\n\nNo relevant information found in your study materials for this topic. You must respond EXACTLY: "I cannot help with that topic as it is not covered in the study materials."\n\nLearner asked: ${message}`;
        }
      } else {
        // Subsequent message - include context if available
        if (studyContext) {
          contextualizedMessage = `=== RELEVANT INFORMATION ===\n${studyContext}\n\n=== END OF INFORMATION ===\n\nREMEMBER:
- Use ONLY this information
- No citations or source mentions
- Clear, natural teacher-like response\n\nQuestion: ${message}\n\nTeach naturally:`;
        } else {
          contextualizedMessage = `No information found. Respond EXACTLY: "I cannot help with that topic as it is not covered in the study materials."\n\nQuestion: ${message}`;
        }
      }

      const result = await this.chat.sendMessageStream(contextualizedMessage);

      let fullText = '';
      for await (const chunk of result.stream) {
        const chunkText = chunk.text();
        fullText += chunkText;
        onChunk(chunkText);
      }

      return fullText;
    } catch (error) {
      console.error('Error sending streaming message to Gemini:', error);
      // Fall back to mock response on API error
      console.log('Falling back to mock streaming response due to API error');
      const mockResponse = this.getMockResponse(message);
      const words = mockResponse.split(' ');
      let chunkText = '';
      for (const word of words) {
        chunkText += word + ' ';
        onChunk(chunkText);
        await new Promise((resolve) => setTimeout(resolve, 50));
      }
      return mockResponse;
    }
  }

  updateConfig(config: Partial<GeminiServiceConfig>): void {
    const previousLanguage = this.config.language;
    this.config = { ...this.config, ...config };
    // Reset chat when language changes to apply new system prompt
    if (config.language && config.language !== previousLanguage) {
      console.log(`Language changed from ${previousLanguage} to ${config.language}, resetting chat`);
      this.chat = null;
      this.languageChanged = true; // Flag to force system prompt on next message
    }
  }

  async resetChat(): Promise<void> {
    this.chat = null;
    await this.initializeChat();
  }

  getConfig(): GeminiServiceConfig {
    return { ...this.config };
  }
}

export default GeminiService;
