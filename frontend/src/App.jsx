import { useState } from "react";
import ChatWindow from "./components/ChatWindow.jsx";
import ChatInput from "./components/ChatInput.jsx";
import { askQuestionStream } from "./api/chatApi.js";

const PRODUCTS = ["All products", "Credit card", "Personal loan", "Savings account", "Money transfer"];

export default function App() {
  const [messages, setMessages] = useState([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [productFilter, setProductFilter] = useState("All products");

  const handleSend = (question) => {
    const filter = productFilter === "All products" ? null : productFilter;

    setMessages((prev) => [
      ...prev,
      { role: "user", content: question },
      { role: "assistant", content: "", sources: [] },
    ]);
    setIsStreaming(true);

    askQuestionStream(question, filter, {
      onSources: (sources) => {
        setMessages((prev) => {
          const next = [...prev];
          next[next.length - 1] = { ...next[next.length - 1], sources };
          return next;
        });
      },
      onToken: (token) => {
        setMessages((prev) => {
          const next = [...prev];
          const last = next[next.length - 1];
          next[next.length - 1] = { ...last, content: last.content + token };
          return next;
        });
      },
      onDone: () => setIsStreaming(false),
      onError: (message) => {
        setMessages((prev) => {
          const next = [...prev];
          next[next.length - 1] = {
            ...next[next.length - 1],
            content: `Something went wrong: ${message}`,
          };
          return next;
        });
        setIsStreaming(false);
      },
    });
  };

  const handleClear = () => setMessages([]);

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">CT</div>
          <div>
            <div className="brand-name">CrediTrust</div>
            <div className="brand-sub">Complaint Intelligence</div>
          </div>
        </div>

        <div className="sidebar-section">
          <label htmlFor="product-filter">Filter by product</label>
          <select
            id="product-filter"
            value={productFilter}
            onChange={(e) => setProductFilter(e.target.value)}
          >
            {PRODUCTS.map((p) => (
              <option key={p} value={p}>{p}</option>
            ))}
          </select>
        </div>

        <button className="clear-button" onClick={handleClear}>
          Clear conversation
        </button>

        <div className="sidebar-footer">
          Grounded in real CFPB complaint narratives. Every answer cites its evidence.
        </div>
      </aside>

      <main className="main-panel">
        <ChatWindow messages={messages} isStreaming={isStreaming} />
        <ChatInput onSend={handleSend} disabled={isStreaming} />
      </main>
    </div>
  );
}
