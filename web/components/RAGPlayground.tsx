"use client";

import { useState } from "react";
import { ragSamples, ragIndex } from "@/lib/rag-stats";

type Source = { domain: string; text: string };
type Result = { question: string; answer: string; sources: Source[] } | null;

const domainColor: Record<string, string> = {
  legal: "var(--series-1)",
  sec: "var(--series-2)",
  general: "var(--series-3)",
};

const domainLabel: Record<string, string> = {
  legal: "Legal",
  sec: "SEC filings",
  general: "General",
};

export function RAGPlayground() {
  const [question, setQuestion] = useState(ragSamples[0].question);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<Result>(null);

  async function ask() {
    if (!question.trim() || loading) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await fetch("/api/complete-rag", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, max_new_tokens: 150 }),
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
          RAG playground
        </h3>
        <p className="text-xs" style={{ color: "var(--text-muted)" }}>
          Ask anything -- a BM25 retriever pulls the top {ragIndex.retrieveK} matching passages
          from a {ragIndex.totalChunks.toLocaleString()}-chunk index of the training corpus, and
          the RAFT model reads them before answering. The retrieved passages are shown below
          the answer so you can judge whether it actually used them, or fell back on memorized
          SFT content.
        </p>
      </div>

      <div className="flex flex-wrap gap-2">
        {ragSamples.map((s) => {
          const isSelected = s.question === question;
          const preview = s.question.length > 40 ? `${s.question.slice(0, 40)}…` : s.question;
          return (
            <button
              key={s.question}
              onClick={() => {
                setQuestion(s.question);
                setResult(null);
                setError(null);
              }}
              title={s.question}
              className="text-xs px-2.5 py-1 rounded-full transition-colors cursor-pointer text-left"
              style={{
                border: "1px solid var(--border-ring)",
                color: isSelected ? "#fff" : "var(--text-secondary)",
                background: isSelected ? domainColor[s.domain] : "transparent",
              }}
            >
              <span
                className="font-medium"
                style={{ color: isSelected ? "#fff" : domainColor[s.domain] }}
              >
                {domainLabel[s.domain] ?? s.domain}:
              </span>{" "}
              {preview}
            </button>
          );
        })}
      </div>

      <textarea
        value={question}
        onChange={(e) => setQuestion(e.target.value)}
        rows={2}
        className="w-full rounded-lg p-3 text-sm resize-none focus:outline-none"
        style={{
          background: "var(--background)",
          border: "1px solid var(--border-ring)",
          color: "var(--text-primary)",
        }}
        placeholder="Ask a question -- legal, SEC filing, or general..."
      />

      <button
        onClick={ask}
        disabled={loading || !question.trim()}
        className="self-start text-sm font-medium px-4 py-2 rounded-lg transition-opacity disabled:opacity-50 cursor-pointer"
        style={{ background: "var(--series-1)", color: "#fff" }}
      >
        {loading ? "Retrieving + generating…" : "Ask the RAG model"}
      </button>

      {error && (
        <p className="text-sm" style={{ color: "var(--status-critical)" }}>
          {error}
        </p>
      )}

      {result && (
        <div className="flex flex-col gap-3">
          <div
            className="rounded-lg p-4 text-sm leading-relaxed flex flex-col gap-1.5"
            style={{ background: "var(--background)", border: "1px solid var(--border-ring)" }}
          >
            <span className="text-xs font-medium uppercase tracking-wide" style={{ color: "var(--series-1)" }}>
              SLM-125M-RAG answer
            </span>
            <span style={{ color: "var(--text-primary)" }}>{result.answer}</span>
          </div>

          <div className="flex flex-col gap-2">
            <span className="text-xs font-medium uppercase tracking-wide" style={{ color: "var(--text-muted)" }}>
              Retrieved context ({result.sources.length} passages)
            </span>
            {result.sources.map((src, i) => (
              <div
                key={i}
                className="rounded-lg p-3 text-xs leading-relaxed flex flex-col gap-1.5"
                style={{ background: "var(--background)", border: "1px dashed var(--border-ring)" }}
              >
                <div className="flex items-center gap-2">
                  <span
                    className="text-[10px] font-medium uppercase tracking-wide px-1.5 py-0.5 rounded-full"
                    style={{ color: "#fff", background: domainColor[src.domain] ?? "var(--text-muted)" }}
                  >
                    passage {i + 1}
                  </span>
                  <span className="text-[10px] uppercase tracking-wide" style={{ color: "var(--text-muted)" }}>
                    {domainLabel[src.domain] ?? src.domain}
                  </span>
                </div>
                <span style={{ color: "var(--text-secondary)" }}>{src.text}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
