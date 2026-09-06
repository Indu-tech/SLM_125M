"use client";

import {
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  CartesianGrid,
} from "recharts";
import { lossCurve, training } from "@/lib/model-stats";

function TooltipContent({
  active,
  payload,
  label,
}: {
  active?: boolean;
  payload?: { value: number }[];
  label?: number;
}) {
  if (!active || !payload?.length) return null;
  return (
    <div
      className="rounded-md px-3 py-2 text-xs shadow-sm"
      style={{
        background: "var(--surface-1)",
        border: "1px solid var(--border-ring)",
        color: "var(--text-primary)",
      }}
    >
      <div className="tabular" style={{ color: "var(--text-muted)" }}>
        step {label}
      </div>
      <div className="tabular font-medium">loss {payload[0].value.toFixed(3)}</div>
    </div>
  );
}

export function LossCurveChart() {
  return (
    <div
      className="rounded-xl p-5"
      style={{ background: "var(--surface-1)", border: "1px solid var(--border-ring)" }}
    >
      <div className="flex items-baseline justify-between mb-3">
        <h3 className="text-sm font-medium" style={{ color: "var(--text-primary)" }}>
          Training loss
        </h3>
        <span className="text-xs tabular" style={{ color: "var(--text-muted)" }}>
          {training.initialTrainLoss.toFixed(2)} &rarr; {training.finalTrainLoss.toFixed(2)}
        </span>
      </div>
      <div style={{ width: "100%", height: 220 }}>
        <ResponsiveContainer>
          <LineChart data={lossCurve} margin={{ top: 8, right: 12, left: -12, bottom: 0 }}>
            <CartesianGrid stroke="var(--gridline)" vertical={false} />
            <XAxis
              dataKey="step"
              tick={{ fontSize: 11, fill: "var(--text-muted)" }}
              stroke="var(--baseline)"
              tickLine={false}
            />
            <YAxis
              tick={{ fontSize: 11, fill: "var(--text-muted)" }}
              stroke="var(--baseline)"
              tickLine={false}
              width={32}
            />
            <Tooltip content={<TooltipContent />} cursor={{ stroke: "var(--baseline)" }} />
            <Line
              type="monotone"
              dataKey="loss"
              stroke="var(--series-1)"
              strokeWidth={2}
              dot={false}
              activeDot={{ r: 4, stroke: "var(--surface-1)", strokeWidth: 2 }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
      <p className="text-xs mt-2" style={{ color: "var(--text-muted)" }}>
        Cross-entropy loss on training batches, logged every 20 steps across both runs
        (checkpoint resume overlap deduplicated).
      </p>
    </div>
  );
}
