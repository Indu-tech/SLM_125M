import { corpusSplit, corpusTotals } from "@/lib/model-stats";

const COLORS = ["var(--series-1)", "var(--series-2)", "var(--series-3)"];

export function CorpusSplitBar() {
  return (
    <div
      className="rounded-xl p-5"
      style={{ background: "var(--surface-1)", border: "1px solid var(--border-ring)" }}
    >
      <div className="flex items-baseline justify-between mb-3">
        <h3 className="text-sm font-medium" style={{ color: "var(--text-primary)" }}>
          Training corpus composition
        </h3>
        <span className="text-xs tabular" style={{ color: "var(--text-muted)" }}>
          {(corpusTotals.trainTokens / 1e9).toFixed(2)}B tokens total
        </span>
      </div>

      <div className="flex w-full h-8 rounded-md overflow-hidden" style={{ gap: "2px" }}>
        {corpusSplit.map((seg, i) => (
          <div
            key={seg.shortName}
            className="flex items-center justify-center h-full"
            style={{
              width: `${seg.pct}%`,
              background: COLORS[i],
              minWidth: seg.pct < 6 ? undefined : 0,
            }}
            title={`${seg.source}: ${seg.pct}% (${(seg.tokens / 1e9).toFixed(3)}B tokens)`}
          >
            <span className="text-xs font-medium px-1 truncate" style={{ color: "#fff" }}>
              {seg.pct.toFixed(1)}%
            </span>
          </div>
        ))}
      </div>

      <div className="flex flex-wrap gap-x-5 gap-y-2 mt-4">
        {corpusSplit.map((seg, i) => (
          <div key={seg.shortName} className="flex items-center gap-2">
            <span
              className="inline-block w-2.5 h-2.5 rounded-sm shrink-0"
              style={{ background: COLORS[i] }}
            />
            <span className="text-xs" style={{ color: "var(--text-secondary)" }}>
              {seg.source}{" "}
              <span className="tabular" style={{ color: "var(--text-muted)" }}>
                ({(seg.tokens / 1e9).toFixed(2)}B tok)
              </span>
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
