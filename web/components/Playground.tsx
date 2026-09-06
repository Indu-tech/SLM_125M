"use client";

import { useState } from "react";
import { samplePrompts } from "@/lib/model-stats";

type Result = { prompt: string; completion: string } | null;

export function Playground() {
  const [prompt, setPrompt] = useState(samplePrompts[0].prompt);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<Result>(null);

  async function generate() {
    if (!prompt.trim() || loading) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await fetch("/api/complete", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt, max_new_tokens: 40 }),
      });
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data?.error || `request failed (${res.status})`);
      }
      setResult(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : "something went wrong");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div
      className="rounded-xl p-5 flex flex-col gap-4"
      style={{ background: "var(--surface-1)", border: "1px solid var(--border-ring)" }}
    >
      <div>
        <h3 className="text-sm font-medium mb-1" style={{ color: "var(--text-primary)" }}>
          Try a completion
        </h3>
        <p className="text-xs" style={{ color: "var(--text-muted)" }}>
          Pick a sentence pulled from the training corpus, or write your own. The model
          continues it &mdash; this is a 125M-parameter model trained for one epoch, so
          expect rough, sometimes funny completions, not polished prose.
        </p>
      </div>

      <div className="flex flex-wrap gap-2">
        {samplePrompts.map((p) => {
          const preview = p.prompt.length > 34 ? `${p.prompt.slice(0, 34)}…` : p.prompt;
          return (
            <button
              key={p.prompt}
              onClick={() => setPrompt(p.prompt)}
              title={p.prompt}
              className="text-xs px-2.5 py-1 rounded-full transition-colors cursor-pointer text-left"
              style={{
                border: "1px solid var(--border-ring)",
                color: prompt === p.prompt ? "#fff" : "var(--text-secondary)",
                background: prompt === p.prompt ? "var(--series-1)" : "transparent",
              }}
            >
              <span
                className="font-medium"
                style={{ color: prompt === p.prompt ? "#fff" : "var(--text-muted)" }}
              >
                {p.source}:
              </span>{" "}
              {preview}
            </button>
          );
        })}
      </div>

      <textarea
        value={prompt}
        onChange={(e) => setPrompt(e.target.value)}
        rows={3}
        className="w-full rounded-lg p-3 text-sm resize-none focus:outline-none"
        style={{
          background: "var(--background)",
          border: "1px solid var(--border-ring)",
          color: "var(--text-primary)",
        }}
        placeholder="Type a sentence for the model to complete..."
      />

      <button
        onClick={generate}
        disabled={loading || !prompt.trim()}
        className="self-start text-sm font-medium px-4 py-2 rounded-lg transition-opacity disabled:opacity-50 cursor-pointer"
        style={{ background: "var(--series-1)", color: "#fff" }}
      >
        {loading ? "Generating…" : "Complete"}
      </button>

      {error && (
        <p className="text-sm" style={{ color: "var(--status-critical)" }}>
          {error}
        </p>
      )}

      {result && (
        <div
          className="rounded-lg p-4 text-sm leading-relaxed"
          style={{ background: "var(--background)", border: "1px solid var(--border-ring)" }}
        >
          <span style={{ color: "var(--text-primary)" }}>{result.prompt}</span>
          <span style={{ color: "var(--text-secondary)" }}>{result.completion}</span>
        </div>
      )}
    </div>
  );
}
