export function StatTile({
  label,
  value,
  sublabel,
}: {
  label: string;
  value: string;
  sublabel?: string;
}) {
  return (
    <div
      className="rounded-xl p-5 flex flex-col gap-1"
      style={{
        background: "var(--surface-1)",
        border: "1px solid var(--border-ring)",
      }}
    >
      <span className="text-xs uppercase tracking-wide" style={{ color: "var(--text-muted)" }}>
        {label}
      </span>
      <span className="text-2xl font-semibold tabular" style={{ color: "var(--text-primary)" }}>
        {value}
      </span>
      {sublabel && (
        <span className="text-xs" style={{ color: "var(--text-secondary)" }}>
          {sublabel}
        </span>
      )}
    </div>
  );
}
