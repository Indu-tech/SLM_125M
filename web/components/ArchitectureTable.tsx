import { architecture } from "@/lib/model-stats";

const rows: [string, string][] = [
  ["Family", architecture.family],
  ["Layers", String(architecture.numLayers)],
  ["Hidden size", String(architecture.hiddenSize)],
  ["Intermediate size (SwiGLU)", String(architecture.intermediateSize)],
  ["Attention heads", `${architecture.numHeads} (MHA, no GQA)`],
  ["Vocab size", architecture.vocabSize.toLocaleString()],
  ["Context length", String(architecture.maxPositionEmbeddings)],
  ["Position encoding", `RoPE (θ=${architecture.ropeTheta.toLocaleString()})`],
  ["Activation", architecture.activation],
  ["Norm", `RMSNorm (eps=${architecture.rmsNormEps})`],
  ["Tied embeddings", architecture.tiedEmbeddings ? "yes" : "no"],
];

export function ArchitectureTable() {
  return (
    <div
      className="rounded-xl p-5"
      style={{ background: "var(--surface-1)", border: "1px solid var(--border-ring)" }}
    >
      <h3 className="text-sm font-medium mb-3" style={{ color: "var(--text-primary)" }}>
        Llama decoder architecture
      </h3>
      <dl className="grid grid-cols-1 sm:grid-cols-2 gap-x-6 gap-y-2">
        {rows.map(([label, value]) => (
          <div key={label} className="flex justify-between gap-4 py-1 text-sm border-b" style={{ borderColor: "var(--gridline)" }}>
            <dt style={{ color: "var(--text-secondary)" }}>{label}</dt>
            <dd className="tabular text-right" style={{ color: "var(--text-primary)" }}>
              {value}
            </dd>
          </div>
        ))}
      </dl>
    </div>
  );
}
