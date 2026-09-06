"use client";

import { useState } from "react";
import { qaSamples, type QASample } from "@/lib/sft-stats";

type Result = { question: string; answer: string } | null;

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

export function QAPlayground() {
  const [selected, setSelected] = useState<QASample>(qaSamples[0]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<Result>(null);

  function select(sample: QASample) {
    setSelected(sample);
    setResult(null);
    setError(null);
  }

  async function ask() {
    if (loading) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await fetch("/api/complete-sft", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: selected.prompt, max_new_tokens: 120 }),
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
          QA playground
        </h3>
        <p className="text-xs" style={{ color: "var(--text-muted)" }}>
          Pick a question drawn from the actual SFT training set. The model answers it
          from scratch &mdash; compare against the reference answer it was trained on.
        </p>
      </div>

      <div className="flex flex-col gap-2">
        {qaSamples.map((s) => {
          const isSelected = s.prompt === selected.prompt;
          return (
            <button
              key={s.prompt}
              onClick={() => select(s)}
              className="text-left text-xs px-3 py-2 rounded-lg transition-colors cursor-pointer flex items-start gap-2"
              style={{
                border: "1px solid var(--border-ring)",
                background: isSelected ? "color-mix(in srgb, var(--series-1) 10%, transparent)" : "transparent",
              }}
            >
              <span
                className="mt-0.5 shrink-0 text-[10px] font-medium uppercase tracking-wide px-1.5 py-0.5 rounded-full"
                style={{
                  color: "#fff",
                  background: domainColor[s.domain] ?? "var(--text-muted)",
                }}
              >
                {domainLabel[s.domain] ?? s.domain}
              </span>
              <span style={{ color: "var(--text-primary)" }}>{s.prompt}</span>
            </button>
          );
        })}
      </div>

      <button
        onClick={ask}
        disabled={loading}
        className="self-start text-sm font-medium px-4 py-2 rounded-lg transition-opacity disabled:opacity-50 cursor-pointer"
        style={{ background: "var(--series-1)", color: "#fff" }}
      >
        {loading ? "Asking the SLM…" : "Get answer from SLM"}
      </button>

      {error && (
        <p className="text-sm" style={{ color: "var(--status-critical)" }}>
          {error}
        </p>
      )}

      {result && (
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          <div
            className="rounded-lg p-4 text-sm leading-relaxed flex flex-col gap-1.5"
            style={{ background: "var(--background)", border: "1px solid var(--border-ring)" }}
          >
            <span className="text-xs font-medium uppercase tracking-wide" style={{ color: "var(--series-1)" }}>
              SLM-125M-SFT answer
            </span>
            <span style={{ color: "var(--text-primary)" }}>{result.answer}</span>
          </div>
          <div
            className="rounded-lg p-4 text-sm leading-relaxed flex flex-col gap-1.5"
            style={{ background: "var(--background)", border: "1px solid var(--border-ring)" }}
          >
            <span className="text-xs font-medium uppercase tracking-wide" style={{ color: "var(--text-muted)" }}>
              Reference (training) answer
            </span>
            <span style={{ color: "var(--text-secondary)" }}>{selected.response}</span>
          </div>
        </div>
      )}
    </div>
  );
}
