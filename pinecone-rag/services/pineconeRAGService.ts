/**
 * Pinecone RAG Service for Frontend
 * =================================
 * TypeScript service for NCERT Q&A using Pinecone backend
 */

export interface PineconeQuery {
  question: string;
  top_k?: number;
}

export interface PineconeSource {
  content: string;
  metadata: {
    class: string;
    subject: string;
  };
  score: number;
}

export interface PineconeResponse {
  answer: string | null;
  sources: PineconeSource[];
  max_score: number;
  context?: string;
  response_time?: number;
}

export class PineconeRAGService {
  private baseUrl: string;

  constructor(baseUrl: string = '/api') {
    this.baseUrl = baseUrl;
  }

  /**
   * Query the NCERT RAG system
   */
  async query(query: PineconeQuery): Promise<PineconeResponse> {
    const startTime = performance.now();

    try {
      const response = await fetch(`${this.baseUrl}/pinecone/query`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(query),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data: PineconeResponse = await response.json();

      // Add response time
      const endTime = performance.now();
      data.response_time = Math.round(endTime - startTime);

      return data;
    } catch (error) {
      console.error('Pinecone RAG Service Error:', error);
      return {
        answer: null,
        sources: [],
        max_score: 0,
        response_time: 0,
      };
    }
  }

  /**
   * Get system status
   */
  async getStatus(): Promise<{
    status: 'connected' | 'disconnected';
    index_stats?: {
      total_vectors: number;
      dimension: number;
      index_fullness: number;
    };
  }> {
    try {
      const response = await fetch(`${this.baseUrl}/pinecone/status`);

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Status check failed:', error);
      return { status: 'disconnected' };
    }
  }

  /**
   * Format response for display
   */
  formatResponse(response: PineconeResponse): {
    answerText: string;
    hasSources: boolean;
    confidence: 'high' | 'medium' | 'low';
    sourcesHtml?: string;
  } {
    const hasSources = response.sources && response.sources.length > 0;

    // Determine confidence based on max_score
    let confidence: 'high' | 'medium' | 'low' = 'low';
    if (response.max_score > 0.7) {
      confidence = 'high';
    } else if (response.max_score > 0.5) {
      confidence = 'medium';
    }

    // Format answer
    let answerText = '';
    if (response.answer) {
      answerText = response.answer;
    } else {
      answerText = hasSources
        ? "I found relevant information, but couldn't generate a complete answer. Please review the sources below."
        : "This information is not available in the NCERT materials for Classes 1-7.";
    }

    // Format sources
    let sourcesHtml = '';
    if (hasSources) {
      sourcesHtml = response.sources.map((source, index) => `
        <div class="ncert-source" style="
          margin: 8px 0;
          padding: 12px;
          background: #f5f5f5;
          border-left: 3px solid #4CAF50;
          border-radius: 4px;
        ">
          <div style="font-weight: bold; color: #333; margin-bottom: 4px;">
            📚 Source ${index + 1}: Class ${source.metadata.class} - ${source.metadata.subject}
          </div>
          <div style="font-size: 0.9em; color: #666; line-height: 1.4;">
            ${source.content.substring(0, 200)}${source.content.length > 200 ? '...' : ''}
          </div>
          <div style="font-size: 0.8em; color: #999; margin-top: 4px;">
            Relevance: ${Math.round(source.score * 100)}%
          </div>
        </div>
      `).join('');
    }

    return {
      answerText,
      hasSources,
      confidence,
      sourcesHtml,
    };
  }

  /**
   * Example questions for testing
   */
  getExampleQuestions(): string[] {
    return [
      "What is photosynthesis?",
      "What are collective nouns? Give examples.",
      "Why do we need to keep our environment clean?",
      "What are the different types of shapes in geometry?",
      "Explain the water cycle in simple words.",
      "What are synonyms and antonyms?",
      "How do plants make their food?",
      "What is a fraction? Explain with examples."
    ];
  }
}

// Export singleton instance
export const pineconeRAG = new PineconeRAGService();