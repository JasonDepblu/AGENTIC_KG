import { User, Bot, AlertCircle, Info, CheckCircle2, RefreshCw } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import type { ChatMessage } from '../../types';

interface MessageProps {
  message: ChatMessage;
}

export function Message({ message }: MessageProps) {
  const { role, content, agentName, isStreaming, isCriticFeedback } = message;

  if (role === 'system') {
    return (
      <div className="flex items-center gap-2 py-2 px-3 rounded-lg bg-bg-secondary/50 text-text-secondary text-sm">
        <Info size={16} />
        <span>{content}</span>
      </div>
    );
  }

  // Special rendering for critic feedback messages
  if (isCriticFeedback) {
    const isValid = content.toLowerCase().trim().startsWith('valid');
    const isRetry = content.toLowerCase().trim().startsWith('retry');

    return (
      <div className={`
        flex items-start gap-3 py-2 px-3 rounded-lg text-sm
        ${isValid
          ? 'bg-green-500/10 border border-green-500/30'
          : isRetry
          ? 'bg-yellow-500/10 border border-yellow-500/30'
          : 'bg-purple-500/10 border border-purple-500/30'
        }
      `}>
        {/* Status icon */}
        <div className="flex-shrink-0 mt-0.5">
          {isValid ? (
            <CheckCircle2 size={16} className="text-green-400" />
          ) : isRetry ? (
            <RefreshCw size={16} className="text-yellow-400" />
          ) : (
            <Bot size={16} className="text-purple-400" />
          )}
        </div>

        {/* Content */}
        <div className="flex-1">
          {/* Agent name */}
          <div className={`text-xs mb-1 ${
            isValid ? 'text-green-400' : isRetry ? 'text-yellow-400' : 'text-purple-400'
          }`}>
            {agentName || 'Critic'} • {isValid ? 'Validated' : isRetry ? 'Needs Revision' : 'Feedback'}
          </div>

          {/* Message content */}
          <div className={`prose prose-sm max-w-none ${
            isValid ? 'text-green-200' : isRetry ? 'text-yellow-200' : 'text-purple-200'
          }`}>
            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
              components={{
                p: ({ children }) => <p className="m-0 mb-1 last:mb-0">{children}</p>,
                ul: ({ children }) => <ul className="m-0 pl-4 list-disc">{children}</ul>,
                li: ({ children }) => <li className="m-0">{children}</li>,
              }}
            >
              {content}
            </ReactMarkdown>
          </div>
        </div>
      </div>
    );
  }

  const isUser = role === 'user';
  const isError = content.startsWith('Error:');

  return (
    <div className={`flex gap-4 ${isUser ? 'flex-row-reverse' : ''}`}>
      {/* Avatar */}
      <div
        className={`
          flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center
          ${isUser ? 'bg-accent' : isError ? 'bg-red-500/20' : 'bg-bg-secondary'}
        `}
      >
        {isUser ? (
          <User size={18} className="text-white" />
        ) : isError ? (
          <AlertCircle size={18} className="text-red-400" />
        ) : (
          <Bot size={18} className="text-accent" />
        )}
      </div>

      {/* Content */}
      <div className={`flex-1 ${isUser ? 'text-right' : ''}`}>
        {/* Agent name */}
        {!isUser && agentName && (
          <div className="text-xs text-text-secondary mb-1">{agentName}</div>
        )}

        {/* Message bubble */}
        <div
          className={`
            inline-block max-w-[85%] rounded-2xl px-4 py-3
            ${isUser
              ? 'bg-accent text-white rounded-tr-sm'
              : isError
              ? 'bg-red-500/10 text-red-400 rounded-tl-sm'
              : 'bg-bg-secondary text-text-primary rounded-tl-sm'
            }
          `}
        >
          <div className={`prose prose-sm max-w-none ${isUser ? 'prose-invert' : 'prose-invert'}`}>
            {isUser ? (
              <p className="m-0">{content}</p>
            ) : (
              <ReactMarkdown
                remarkPlugins={[remarkGfm]}
                components={{
                  p: ({ children }) => <p className="m-0 mb-2 last:mb-0">{children}</p>,
                  code: ({ children, className }) => {
                    const isInline = !className;
                    return isInline ? (
                      <code className="bg-black/20 px-1 py-0.5 rounded text-sm">{children}</code>
                    ) : (
                      <code className="block bg-black/20 p-3 rounded-lg text-sm overflow-x-auto">
                        {children}
                      </code>
                    );
                  },
                  pre: ({ children }) => <pre className="m-0">{children}</pre>,
                  ul: ({ children }) => <ul className="m-0 pl-4 list-disc">{children}</ul>,
                  ol: ({ children }) => <ol className="m-0 pl-4 list-decimal">{children}</ol>,
                  li: ({ children }) => <li className="m-0">{children}</li>,
                  // Table components for schema display
                  table: ({ children }) => (
                    <div className="overflow-x-auto my-3">
                      <table className="min-w-full border-collapse border border-border/50 rounded-lg overflow-hidden text-sm">
                        {children}
                      </table>
                    </div>
                  ),
                  thead: ({ children }) => (
                    <thead className="bg-accent/20">{children}</thead>
                  ),
                  tbody: ({ children }) => (
                    <tbody className="divide-y divide-border/30">{children}</tbody>
                  ),
                  tr: ({ children }) => (
                    <tr className="hover:bg-white/5 transition-colors">{children}</tr>
                  ),
                  th: ({ children }) => (
                    <th className="px-3 py-2 text-left font-semibold text-accent border-b border-border/50">
                      {children}
                    </th>
                  ),
                  td: ({ children }) => (
                    <td className="px-3 py-2 text-text-primary border-b border-border/20">
                      {children}
                    </td>
                  ),
                }}
              >
                {content}
              </ReactMarkdown>
            )}
          </div>

          {/* Streaming indicator */}
          {isStreaming && (
            <span className="inline-block w-2 h-4 bg-accent ml-1 animate-pulse" />
          )}
        </div>
      </div>
    </div>
  );
}
