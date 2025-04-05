"use client";

import { useState, useRef, useEffect } from 'react';
import { PaperChatMessage } from '../lib/types';

interface PaperChatProps {
  paperSlug: string;
  paperTitle: string;
}

export function PaperChat({ paperSlug, paperTitle }: PaperChatProps) {
  const [messages, setMessages] = useState<PaperChatMessage[]>([
    { role: 'system', content: `You are a helpful assistant discussing the paper "${paperTitle}". Answer questions based only on the paper content.` }
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom of messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    // Add user message
    const userMessage: PaperChatMessage = { role: 'user', content: input };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const response = await fetch('/api/paper-chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: [...messages, userMessage],
          paperSlug
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to get response');
      }

      const data = await response.json();
      setMessages(prev => [...prev, data.message]);
    } catch (error) {
      console.error('Error getting AI response:', error);
      setMessages(prev => [
        ...prev,
        { role: 'assistant', content: 'Sorry, I encountered an error processing your request.' }
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="border border-neutral-200 dark:border-neutral-800 rounded-lg p-4 my-8">
      <h3 className="font-medium text-lg mb-4">Chat about this paper</h3>
      
      <div className="max-h-80 overflow-y-auto mb-4 space-y-4">
        {messages.filter(m => m.role !== 'system').map((message, index) => (
          <div 
            key={index} 
            className={`p-3 rounded-lg ${
              message.role === 'user' 
                ? 'bg-blue-100 dark:bg-blue-900 ml-8' 
                : 'bg-neutral-100 dark:bg-neutral-800 mr-8'
            }`}
          >
            <p className="text-sm font-medium mb-1">
              {message.role === 'user' ? 'You' : 'AI Assistant'}
            </p>
            <p className="text-sm whitespace-pre-wrap">{message.content}</p>
          </div>
        ))}
        {isLoading && (
          <div className="bg-neutral-100 dark:bg-neutral-800 p-3 rounded-lg mr-8">
            <p className="text-sm font-medium mb-1">AI Assistant</p>
            <p className="text-sm">Thinking...</p>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>
      
      <form onSubmit={handleSubmit} className="flex items-end gap-2">
        <div className="flex-grow">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask a question about this paper..."
            className="w-full p-2 border border-neutral-200 dark:border-neutral-700 rounded-md bg-transparent"
            rows={1}
            disabled={isLoading}
          />
        </div>
        <button
          type="submit"
          disabled={isLoading || !input.trim()}
          className="bg-blue-500 hover:bg-blue-600 text-white py-2 px-4 rounded-md disabled:opacity-50 disabled:cursor-not-allowed text-sm"
        >
          Send
        </button>
      </form>
    </div>
  );
} 