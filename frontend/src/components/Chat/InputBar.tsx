import { useState, useRef, useEffect } from 'react';
import { Send, Paperclip, Loader2 } from 'lucide-react';

interface InputBarProps {
  onSend: (content: string) => void;
  disabled?: boolean;
  isLoading?: boolean;
}

export function InputBar({ onSend, disabled, isLoading }: InputBarProps) {
  const [value, setValue] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Auto-resize textarea
  useEffect(() => {
    const textarea = textareaRef.current;
    if (textarea) {
      textarea.style.height = 'auto';
      textarea.style.height = `${Math.min(textarea.scrollHeight, 200)}px`;
    }
  }, [value]);

  const handleSubmit = () => {
    console.log('handleSubmit called, value:', value, 'disabled:', disabled);
    if (value.trim() && !disabled) {
      console.log('Sending message:', value.trim());
      onSend(value.trim());
      setValue('');
    } else {
      console.log('Message not sent - empty or disabled');
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <div className="max-w-3xl mx-auto px-6 py-4">
      <div className="relative flex items-end gap-2 bg-bg-input rounded-2xl border border-border focus-within:border-accent/50 transition-colors">
        {/* Attachment button */}
        <button
          className="p-3 text-text-secondary hover:text-text-primary transition-colors"
          title="Attach file"
          disabled={disabled}
        >
          <Paperclip size={20} />
        </button>

        {/* Text input */}
        <textarea
          ref={textareaRef}
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Describe the knowledge graph you want to build..."
          disabled={disabled}
          rows={1}
          className="
            flex-1 bg-transparent py-3 px-1 text-text-primary placeholder-text-secondary
            resize-none outline-none max-h-[200px]
          "
        />

        {/* Send/Stop button */}
        <button
          onClick={handleSubmit}
          disabled={disabled || !value.trim()}
          className={`
            p-3 rounded-xl transition-colors
            ${value.trim() && !disabled
              ? 'bg-accent text-white hover:bg-accent-hover'
              : 'text-text-secondary'
            }
          `}
          title={isLoading ? 'Stop' : 'Send message'}
        >
          {isLoading ? (
            <Loader2 size={20} className="animate-spin" />
          ) : (
            <Send size={20} />
          )}
        </button>
      </div>

      {/* Footer hint */}
      <p className="text-xs text-text-secondary text-center mt-2">
        Press Enter to send, Shift+Enter for new line
      </p>
    </div>
  );
}
