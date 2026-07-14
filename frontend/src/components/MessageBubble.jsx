import SourcesPanel from "./SourcesPanel.jsx";

export default function MessageBubble({ role, content, sources, isStreaming }) {
  const isUser = role === "user";

  return (
    <div className={`message-row ${isUser ? "message-row--user" : "message-row--assistant"}`}>
      <div className={`message-bubble ${isUser ? "message-bubble--user" : "message-bubble--assistant"}`}>
        {!isUser && <div className="message-label">Analyst</div>}
        <p className="message-content">
          {content}
          {isStreaming && <span className="cursor-blink">▍</span>}
        </p>
        {!isUser && !isStreaming && <SourcesPanel sources={sources} />}
      </div>
    </div>
  );
}
