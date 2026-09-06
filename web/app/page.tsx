import Link from "next/link";
import { StatTile } from "@/components/StatTile";
import { CorpusSplitBar } from "@/components/CorpusSplitBar";
import { LossCurveChart } from "@/components/LossCurveChart";
import { ArchitectureTable } from "@/components/ArchitectureTable";
import { Playground } from "@/components/Playground";
import { architecture, training, cost, corpusTotals, modelMeta } from "@/lib/model-stats";

function fmtCompact(n: number): string {
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)}B`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}K`;
  return String(n);
}

export default function Home() {
  return (
    <div className="min-h-full flex flex-col">
      <main className="flex-1 w-full max-w-5xl mx-auto px-4 sm:px-6 py-12 flex flex-col gap-10">
        <header className="flex flex-col gap-3">
          <div className="flex items-center gap-2">
            <span
              className="text-xs font-medium tracking-wide uppercase w-fit px-2 py-1 rounded-full"
              style={{ color: "var(--series-1)", background: "color-mix(in srgb, var(--series-1) 12%, transparent)" }}
            >
              Built from scratch
            </span>
            <Link href="/sft" className="text-xs underline" style={{ color: "var(--text-muted)" }}>
              fine-tuned SFT model &rarr;
            </Link>
            <Link href="/rag" className="text-xs underline" style={{ color: "var(--text-muted)" }}>
              RAFT + RAG model &rarr;
            </Link>
          </div>
          <h1 className="text-3xl sm:text-4xl font-semibold" style={{ color: "var(--text-primary)" }}>
            slm-125m: a 125M-parameter language model, pretrained from scratch
          </h1>
          <p className="max-w-2xl text-sm sm:text-base" style={{ color: "var(--text-secondary)" }}>
            A Llama-architecture decoder trained on a legal + financial + web corpus, entirely
            on Modal (data pipeline &amp; training) with weights hosted on Hugging Face.
            {modelMeta.hfPrivate && " The Hub repo is private; this page is the public window into it."}
          </p>
        </header>

        <section className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-4">
          <StatTile label="Trainable parameters" value={fmtCompact(architecture.paramsTotal)} />
          <StatTile
            label="Training tokens"
            value={fmtCompact(training.tokensTrained)}
            sublabel={`${training.totalSteps.toLocaleString()} steps · 1 epoch`}
          />
          <StatTile label="Epochs" value={String(training.epochs)} sublabel="completed across 2 runs" />
          <StatTile
            label="Compute cost"
            value={`$${cost.totalUsd.toFixed(2)}`}
            sublabel={`8x H100 · target was $${cost.budgetTargetUsd}`}
          />
          <StatTile label="Corpus (unique)" value={fmtCompact(corpusTotals.trainTokens)} sublabel="3 sources, deduped" />
          <StatTile label="Held-out val set" value={corpusTotals.valWindows.toLocaleString()} sublabel="1024-token windows" />
          <StatTile label="Context length" value={`${architecture.maxPositionEmbeddings}`} sublabel="tokens" />
          <StatTile label="Vocab size" value={architecture.vocabSize.toLocaleString()} sublabel="byte-level BPE" />
        </section>

        <section className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <ArchitectureTable />
          <div className="flex flex-col gap-4">
            <CorpusSplitBar />
            <LossCurveChart />
          </div>
        </section>

        <section>
          <Playground />
        </section>

        <footer
          className="text-xs pt-6 flex flex-col sm:flex-row sm:justify-between gap-2"
          style={{ color: "var(--text-muted)", borderTop: "1px solid var(--gridline)" }}
        >
          <p>
            Weights + tokenizer:{" "}
            <a
              className="underline"
              href={modelMeta.hfUrl}
              target="_blank"
              rel="noreferrer"
              style={{ color: "var(--text-secondary)" }}
            >
              {modelMeta.hfRepo}
            </a>{" "}
            (private)
          </p>
          <p>Data pipeline, training &amp; inference on Modal · frontend on Vercel</p>
        </footer>
      </main>
    </div>
  );
}
