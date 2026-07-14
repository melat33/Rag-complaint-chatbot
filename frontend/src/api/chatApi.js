const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

/**
 * Single-shot ask: waits for the full answer + sources.
 */
export async function askQuestion(question, productFilter = null) {
  const res = await fetch(`${API_BASE}/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, product_filter: productFilter }),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Request failed" }));
    throw new Error(err.detail || "Request failed");
  }
  return res.json(); // { answer, sources }
}

/**
 * Streaming ask: calls onSources once, onToken repeatedly, onDone when finished.
 * Uses the Fetch API + ReadableStream to consume Server-Sent Events, since
 * EventSource doesn't support POST bodies.
 */
export async function askQuestionStream(question, productFilter, { onSources, onToken, onDone, onError }) {
  try {
    const res = await fetch(`${API_BASE}/ask/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, product_filter: productFilter }),
    });

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n\n");
      buffer = lines.pop(); // keep incomplete event in buffer

      for (const line of lines) {
        if (!line.startsWith("data: ")) continue;
        const event = JSON.parse(line.slice(6));

        if (event.type === "sources") onSources?.(event.sources);
        else if (event.type === "token") onToken?.(event.token);
        else if (event.type === "error") onError?.(event.message);
        else if (event.type === "done") onDone?.();
      }
    }
  } catch (err) {
    onError?.(err.message);
  }
}

export async function checkHealth() {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) throw new Error("Backend unavailable");
  return res.json();
}
