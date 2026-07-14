import { useState } from "react";

/**
 * Displays the retrieved chunks an answer was grounded in. This is the
 * "trust and verification" requirement from Task 4 - a PM should be able
 * to click through and confirm the AI isn't fabricating a trend.
 */
export default function SourcesPanel({ sources }) {
  const [open, setOpen] = useState(false);

  if (!sources || sources.length === 0) return null;

  return (
    <div className="sources-panel">
      <button className="sources-toggle" onClick={() => setOpen((o) => !o)}>
        <span className="sources-toggle-icon">{open ? "−" : "+"}</span>
        Evidence · {sources.length} excerpt{sources.length > 1 ? "s" : ""}
      </button>

      {open && (
        <div className="sources-list">
          {sources.map((s, i) => {
            const similarity = Math.max(0, Math.round((1 - s.distance) * 100));
            return (
              <div className="evidence-card" key={i}>
                <div className="evidence-stamp">{similarity}% match</div>
                <p className="evidence-text">&ldquo;{s.text}&rdquo;</p>
                <div className="evidence-meta">
                  <span>#{s.complaint_id}</span>
                  {s.product_category && <span>{s.product_category}</span>}
                  {s.issue && <span>{s.issue}</span>}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
