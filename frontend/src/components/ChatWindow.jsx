import { useEffect, useRef } from "react";
import MessageBubble from "./MessageBubble.jsx";

export default function ChatWindow({ messages, isStreaming }) {
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  if (messages.length === 0) {
    return (
      <div className="chat-empty">
        <div className="chat-empty-eyebrow">Complaint Intelligence</div>
        <h1>Ask a question.</h1>
        <p>
          Query customer complaints across Credit Cards, Personal Loans, Savings Accounts,
          and Money Transfers
        </p>
      </div>
    );
  }

  return (
    <div className="chat-window">
      {messages.map((m, i) => (
        <MessageBubble
          key={i}
          role={m.role}
          content={m.content}
          sources={m.sources}
          isStreaming={isStreaming && i === messages.length - 1 && m.role === "assistant"}
        />
      ))}
      <div ref={bottomRef} />
    </div>
  );
}
