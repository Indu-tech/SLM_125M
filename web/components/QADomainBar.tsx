import { qaDataset } from "@/lib/sft-stats";

const COLORS = ["var(--series-1)", "var(--series-2)", "var(--series-3)"];

export function QADomainBar() {
  return (
    <div
      className="rounded-xl p-5"
      style={{ background: "var(--surface-1)", border: "1px solid var(--border-ring)" }}
    >
      <div className="flex items-baseline justify-between mb-3">
        <h3 className="text-sm font-medium" style={{ color: "var(--text-primary)" }}>
          SFT dataset composition
        </h3>
        <span className="text-xs tabular" style={{ color: "var(--text-muted)" }}>
          {qaDataset.totalPairs.toLocaleString()} QA pairs total
        </span>
      </div>

      <div className="flex w-full h-8 rounded-md overflow-hidden" style={{ gap: "2px" }}>
        {qaDataset.domainCounts.map((seg, i) => {
          const pct = (seg.count / qaDataset.totalPairs) * 100;
          return (
            <div
              key={seg.key}
              className="flex items-center justify-center h-full"
              style={{ width: `${pct}%`, background: COLORS[i] }}
              title={`${seg.domain}: ${pct.toFixed(1)}% (${seg.count} pairs)`}
            >
              <span className="text-xs font-medium px-1 truncate" style={{ color: "#fff" }}>
                {pct.toFixed(0)}%
              </span>
            </div>
          );
        })}
      </div>

      <div className="flex flex-wrap gap-x-5 gap-y-2 mt-4">
        {qaDataset.domainCounts.map((seg, i) => (
          <div key={seg.key} className="flex items-center gap-2">
            <span
              className="inline-block w-2.5 h-2.5 rounded-sm shrink-0"
              style={{ background: COLORS[i] }}
            />
            <span className="text-xs" style={{ color: "var(--text-secondary)" }}>
              {seg.domain}{" "}
              <span className="tabular" style={{ color: "var(--text-muted)" }}>
                ({seg.count} pairs)
              </span>
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
