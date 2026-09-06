import Link from "next/link";
import { StatTile } from "@/components/StatTile";
import { RAGDatasetBar } from "@/components/RAGDatasetBar";
import { RAGPlayground } from "@/components/RAGPlayground";
import { raftTraining, raftDataset, raftPerplexity, cost, ragIndex, raftModelMeta } from "@/lib/rag-stats";

export default function RAGPage() {
  return (
    <div className="min-h-full flex flex-col">
      <main className="flex-1 w-full max-w-5xl mx-auto px-4 sm:px-6 py-12 flex flex-col gap-10">
        <header className="flex flex-col gap-3">
          <div className="flex items-center gap-2 flex-wrap">
            <span
              className="text-xs font-medium tracking-wide uppercase w-fit px-2 py-1 rounded-full"
              style={{ color: "var(--series-3)", background: "color-mix(in srgb, var(--series-3) 12%, transparent)" }}
            >
              RAFT + RAG
            </span>
            <Link href="/" className="text-xs underline" style={{ color: "var(--text-muted)" }}>
              &larr; base model page
            </Link>
            <Link href="/sft" className="text-xs underline" style={{ color: "var(--text-muted)" }}>
              &larr; SFT model page
            </Link>
          </div>
          <h1 className="text-3xl sm:text-4xl font-semibold" style={{ color: "var(--text-primary)" }}>
            SLM-125M-RAG model
          </h1>
          <p className="max-w-2xl text-sm sm:text-base" style={{ color: "var(--text-secondary)" }}>
            The SFT model, continued-trained with RAFT (Retrieval-Augmented Fine-Tuning) on
            the same {raftDataset.totalExamples.toLocaleString()} QA pairs reformatted with retrieved-style
            passages, then paired at inference with a BM25 retriever over a {ragIndex.totalChunks.toLocaleString()}-chunk
            index &mdash; entirely on Modal, with weights hosted on Hugging Face.
            {raftModelMeta.hfPrivate && " The Hub repo is private; this page is the public window into it."}
          </p>
        </header>

        <section className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-4">
          <StatTile
            label="Perplexity"
            value={raftPerplexity.value.toFixed(2)}
            sublabel="on assistant-response tokens (train set)"
          />
          <StatTile label="Epochs" value={String(raftTraining.epochs)} sublabel={`${raftTraining.steps} steps · ${raftTraining.gpuType}`} />
          <StatTile
            label="Training examples"
            value={raftDataset.totalExamples.toLocaleString()}
            sublabel="same QA pairs as SFT, reformatted"
          />
          <StatTile
            label="Compute cost"
            value={`$${cost.totalUsd.toFixed(2)}`}
            sublabel={`$${cost.raftTrainUsd.toFixed(3)} measured GPU + CPU est.`}
          />
          <StatTile label="Learning rate" value={raftTraining.lr.toExponential(0)} sublabel={raftTraining.optimizer} />
        </section>

        <section className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <RAGDatasetBar />
          <div
            className="rounded-xl p-5 flex flex-col gap-3"
            style={{ background: "var(--surface-1)", border: "1px solid var(--border-ring)" }}
          >
            <h3 className="text-sm font-medium" style={{ color: "var(--text-primary)" }}>
              RAG retrieval index
            </h3>
            <div className="flex flex-col gap-2">
              <div className="flex items-center justify-between">
                <span className="text-xs" style={{ color: "var(--text-secondary)" }}>Method</span>
                <span className="text-xs tabular" style={{ color: "var(--text-primary)" }}>{ragIndex.method}</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-xs" style={{ color: "var(--text-secondary)" }}>Indexed chunks</span>
                <span className="text-xs tabular" style={{ color: "var(--text-primary)" }}>
                  {ragIndex.totalChunks.toLocaleString()} ({ragIndex.uniqueTerms.toLocaleString()} unique terms)
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-xs" style={{ color: "var(--text-secondary)" }}>Retrieved per query</span>
                <span className="text-xs tabular" style={{ color: "var(--text-primary)" }}>
                  top-{ragIndex.retrieveK} passages
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-xs" style={{ color: "var(--text-secondary)" }}>Domain split</span>
                <span className="text-xs tabular" style={{ color: "var(--text-primary)" }}>
                  {ragIndex.domainCounts.map((d) => `${d.domain} ${((d.count / ragIndex.totalChunks) * 100).toFixed(0)}%`).join(" · ")}
                </span>
              </div>
            </div>
            <p className="text-xs" style={{ color: "var(--text-muted)" }}>{ragIndex.note}</p>
          </div>
        </section>

        <section>
          <RAGPlayground />
        </section>

        <footer
          className="text-xs pt-6 flex flex-col sm:flex-row sm:justify-between gap-2"
          style={{ color: "var(--text-muted)", borderTop: "1px solid var(--gridline)" }}
        >
          <p>
            Weights + tokenizer:{" "}
            <a
              className="underline"
              href={raftModelMeta.hfUrl}
              target="_blank"
              rel="noreferrer"
              style={{ color: "var(--text-secondary)" }}
            >
              {raftModelMeta.hfRepo}
            </a>{" "}
            (private) &middot; continue-trained from{" "}
            <a
              className="underline"
              href={`https://huggingface.co/${raftModelMeta.baseRepo}`}
              target="_blank"
              rel="noreferrer"
              style={{ color: "var(--text-secondary)" }}
            >
              {raftModelMeta.baseRepo}
            </a>
          </p>
          <p>Data pipeline, training, retrieval index &amp; inference on Modal &middot; frontend on Vercel</p>
        </footer>
      </main>
    </div>
  );
}
