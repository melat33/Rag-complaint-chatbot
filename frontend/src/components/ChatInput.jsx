import { useState } from "react";

const SUGGESTED_QUESTIONS = [
  "Why are people unhappy with Credit Cards?",
  "What are the most common Personal Loan complaints?",
  "Are there recurring fraud signals in Money Transfers?",
];

export default function ChatInput({ onSend, disabled }) {
  const [value, setValue] = useState("");

  const submit = (text) => {
    const trimmed = text.trim();
    if (!trimmed || disabled) return;
    onSend(trimmed);
    setValue("");
  };

  return (
    <div className="chat-input-area">
      {value === "" && (
        <div className="suggestions">
          {SUGGESTED_QUESTIONS.map((q) => (
            <button key={q} className="suggestion-chip" onClick={() => submit(q)} disabled={disabled}>
              {q}
            </button>
          ))}
        </div>
      )}
      <form
        className="chat-input-form"
        onSubmit={(e) => {
          e.preventDefault();
          submit(value);
        }}
      >
        <input
          type="text"
          value={value}
          onChange={(e) => setValue(e.target.value)}
          placeholder="Ask about complaint trends…"
          disabled={disabled}
          aria-label="Ask a question about customer complaints"
        />
        <button type="submit" disabled={disabled || !value.trim()} aria-label="Ask">
          Ask
        </button>
      </form>
    </div>
  );
}
