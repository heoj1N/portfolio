import { NextRequest, NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { PaperChatMessage } from '../../lib/types';

// Simple vector similarity function
function cosineSimilarity(a: number[], b: number[]): number {
  let dotProduct = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    dotProduct += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

// Very simple tokenization 
function tokenize(text: string): string[] {
  return text.toLowerCase()
    .replace(/[^\w\s]/g, '')
    .split(/\s+/)
    .filter(Boolean);
}

// Simple TF-IDF implementation
function createEmbedding(text: string): number[] {
  const tokens = tokenize(text);
  const uniqueTokens = Array.from(new Set(tokens));
  
  // Create a vector with term frequencies
  const vector = uniqueTokens.map(token => {
    const count = tokens.filter(t => t === token).length;
    // Simple TF (Term Frequency)
    return count / tokens.length;
  });
  
  return vector;
}

// Simple function to generate a response based on the paper content
function generateResponse(userQuery: string, paperContent: string): string {
  // This is a placeholder for a real LLM
  // In a real implementation, you'd call an actual LLM API
  
  const queryEmbedding = createEmbedding(userQuery);
  
  // Split the paper content into paragraphs
  const paragraphs = paperContent.split('\n\n');
  
  // Find the most relevant paragraph
  let bestScore = -1;
  let mostRelevantParagraph = '';
  
  for (const paragraph of paragraphs) {
    if (paragraph.trim().length < 10) continue;
    
    const embedding = createEmbedding(paragraph);
    // Ensure vectors have the same dimension for comparison
    const minLength = Math.min(queryEmbedding.length, embedding.length);
    const score = cosineSimilarity(
      queryEmbedding.slice(0, minLength), 
      embedding.slice(0, minLength)
    );
    
    if (score > bestScore) {
      bestScore = score;
      mostRelevantParagraph = paragraph;
    }
  }
  
  // Simple response generation
  if (bestScore > 0.1) {
    return `Based on the paper, I found this relevant information:\n\n${mostRelevantParagraph}\n\nThis is a simplified AI response. In a production environment, this would use a real LLM API to generate more contextual and helpful responses.`;
  } else {
    return "I couldn't find specific information about that in the paper. Please try asking about a different aspect of the paper's content. Note: This is using a simplified AI model for demonstration purposes.";
  }
}

export async function POST(request: NextRequest) {
  try {
    const { messages, paperSlug } = await request.json();
    
    if (!messages || !Array.isArray(messages) || !paperSlug) {
      return NextResponse.json(
        { error: 'Invalid request. Messages and paperSlug are required.' },
        { status: 400 }
      );
    }
    
    // Get user's query (last message)
    const userMessage = messages.filter(m => m.role === 'user').pop();
    if (!userMessage) {
      return NextResponse.json(
        { error: 'No user message found.' },
        { status: 400 }
      );
    }
    
    // Get paper content
    const paperPath = path.join(process.cwd(), 'app', 'papers', 'documents', `${paperSlug}.mdx`);
    
    if (!fs.existsSync(paperPath)) {
      return NextResponse.json(
        { error: 'Paper not found.' },
        { status: 404 }
      );
    }
    
    const fileContent = fs.readFileSync(paperPath, 'utf-8');
    // Remove frontmatter
    const content = fileContent.replace(/---\s*([\s\S]*?)\s*---/, '').trim();
    
    // Generate AI response
    const responseText = generateResponse(userMessage.content, content);
    
    const assistantMessage: PaperChatMessage = {
      role: 'assistant',
      content: responseText
    };
    
    // In a production app, you would likely call a real LLM API here
    // with the paper content as context
    
    return NextResponse.json({ message: assistantMessage });
  } catch (error) {
    console.error('Error in paper-chat API:', error);
    return NextResponse.json(
      { error: 'Failed to process request.' },
      { status: 500 }
    );
  }
} 