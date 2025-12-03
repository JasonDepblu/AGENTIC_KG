import { useEffect, useRef } from 'react';
import { useChatStore } from '../../stores/chatStore';
import { Message } from './Message';

export function MessageList() {
  const { messages } = useChatStore();
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  if (messages.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-center px-6">
        <div className="max-w-md">
          <h2 className="text-2xl font-semibold text-text-primary mb-4">
            Build Your Knowledge Graph
          </h2>
          <p className="text-text-secondary mb-6">
            Describe the knowledge graph you want to create. I'll help you analyze your data files
            and construct a graph in Neo4j.
          </p>
          <div className="grid gap-3 text-left">
            <ExamplePrompt text="I want to build a supply chain graph to analyze bill of materials relationships" />
            <ExamplePrompt text="Create a social network graph from my user and friendship data" />
            <ExamplePrompt text="Build a knowledge graph for product recommendations" />
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="h-full overflow-y-auto px-6 py-4">
      <div className="max-w-3xl mx-auto space-y-4">
        {messages.map((message) => (
          <Message key={message.id} message={message} />
        ))}
        <div ref={bottomRef} />
      </div>
    </div>
  );
}

function ExamplePrompt({ text }: { text: string }) {
  return (
    <div className="p-3 rounded-lg bg-bg-secondary border border-border hover:border-accent/50 cursor-pointer transition-colors">
      <p className="text-sm text-text-primary">{text}</p>
    </div>
  );
}
