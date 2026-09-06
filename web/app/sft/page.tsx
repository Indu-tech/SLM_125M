import Link from "next/link";
import { StatTile } from "@/components/StatTile";
import { QADomainBar } from "@/components/QADomainBar";
import { QAPlayground } from "@/components/QAPlayground";
import { sftPerplexity, qaDataset, sftTraining, sftModelMeta, caseholdComparison } from "@/lib/sft-stats";

export default function SFTPage() {
  return (
    <div className="min-h-full flex flex-col">
      <main className="flex-1 w-full max-w-5xl mx-auto px-4 sm:px-6 py-12 flex flex-col gap-10">
        <header className="flex flex-col gap-3">
          <div className="flex items-center gap-2">
            <span
              className="text-xs font-medium tracking-wide uppercase w-fit px-2 py-1 rounded-full"
              style={{ color: "var(--series-2)", background: "color-mix(in srgb, var(--series-2) 12%, transparent)" }}
            >
              Instruction fine-tuned
            </span>
            <Link
              href="/"
              className="text-xs underline"
              style={{ color: "var(--text-muted)" }}
            >
              &larr; base model page
            </Link>
            <Link
              href="/rag"
              className="text-xs underline"
              style={{ color: "var(--text-muted)" }}
            >
              RAFT + RAG model &rarr;
            </Link>
          </div>
          <h1 className="text-3xl sm:text-4xl font-semibold" style={{ color: "var(--text-primary)" }}>
            SLM-125M-SFT model
          </h1>
          <p className="max-w-2xl text-sm sm:text-base" style={{ color: "var(--text-secondary)" }}>
            The pretrained slm-125m base model, further fine-tuned on {qaDataset.totalPairs.toLocaleString()}{" "}
            question-answer pairs grounded in its own legal, SEC filing, and general-knowledge training
            corpus &mdash; entirely on Modal, with weights hosted on Hugging Face.
            {sftModelMeta.hfPrivate && " The Hub repo is private; this page is the public window into it."}
          </p>
        </header>

        <section className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-4">
          <StatTile
            label="Perplexity"
            value={sftPerplexity.value.toFixed(2)}
            sublabel="on assistant-response tokens (train set)"
          />
          <StatTile
            label="QA pairs trained on"
            value={qaDataset.totalPairs.toLocaleString()}
            sublabel="3 domains, LLM-generated"
          />
          <StatTile label="Epochs" value={String(sftTraining.epochs)} sublabel={sftTraining.gpuType} />
          <StatTile label="Learning rate" value={sftTraining.lr.toExponential(0)} sublabel={sftTraining.optimizer} />
        </section>

        <section className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <QADomainBar />
          <div
            className="rounded-xl p-5 flex flex-col gap-3"
            style={{ background: "var(--surface-1)", border: "1px solid var(--border-ring)" }}
          >
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-medium" style={{ color: "var(--text-primary)" }}>
                Base &rarr; SFT comparison score
              </h3>
              <span className="text-xs" style={{ color: "var(--text-muted)" }}>
                {caseholdComparison.task}
              </span>
            </div>

            <div className="grid grid-cols-3 gap-3">
              <div className="flex flex-col gap-0.5">
                <span className="text-xs uppercase tracking-wide" style={{ color: "var(--text-muted)" }}>
                  Base acc_norm
                </span>
                <span className="text-xl font-semibold tabular" style={{ color: "var(--text-primary)" }}>
                  {(caseholdComparison.base.accNorm * 100).toFixed(1)}%
                </span>
              </div>
              <div className="flex flex-col gap-0.5">
                <span className="text-xs uppercase tracking-wide" style={{ color: "var(--text-muted)" }}>
                  SFT acc_norm
                </span>
                <span className="text-xl font-semibold tabular" style={{ color: "var(--series-2)" }}>
                  {(caseholdComparison.sft.accNorm * 100).toFixed(1)}%
                </span>
              </div>
              <div className="flex flex-col gap-0.5">
                <span className="text-xs uppercase tracking-wide" style={{ color: "var(--text-muted)" }}>
                  Delta (SFT&minus;Base)
                </span>
                <span className="text-xl font-semibold tabular" style={{ color: "var(--text-primary)" }}>
                  {caseholdComparison.deltaAccNorm >= 0 ? "+" : ""}
                  {(caseholdComparison.deltaAccNorm * 100).toFixed(1)}pp
                </span>
              </div>
            </div>

            <p className="text-xs" style={{ color: "var(--text-muted)" }}>
              {caseholdComparison.nExamples.toLocaleString()} held-out examples &middot; random baseline{" "}
              {(caseholdComparison.randomBaseline * 100).toFixed(0)}% &middot; {caseholdComparison.note}
            </p>
          </div>
        </section>

        <section>
          <QAPlayground />
        </section>

        <footer
          className="text-xs pt-6 flex flex-col sm:flex-row sm:justify-between gap-2"
          style={{ color: "var(--text-muted)", borderTop: "1px solid var(--gridline)" }}
        >
          <p>
            Weights + tokenizer:{" "}
            <a
              className="underline"
              href={sftModelMeta.hfUrl}
              target="_blank"
              rel="noreferrer"
              style={{ color: "var(--text-secondary)" }}
            >
              {sftModelMeta.hfRepo}
            </a>{" "}
            (private) &middot; fine-tuned from{" "}
            <a
              className="underline"
              href={`https://huggingface.co/${sftModelMeta.baseRepo}`}
              target="_blank"
              rel="noreferrer"
              style={{ color: "var(--text-secondary)" }}
            >
              {sftModelMeta.baseRepo}
            </a>
          </p>
          <p>Data pipeline, training &amp; inference on Modal &middot; frontend on Vercel</p>
        </footer>
      </main>
    </div>
  );
}
