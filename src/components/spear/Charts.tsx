"use client";

interface Series {
  name: string;
  color: string;
  points: { x: number; y: number }[];
}

function buildPath(points: { x: number; y: number }[], xMin: number, xMax: number, yMin: number, yMax: number, w: number, h: number, pad: number) {
  const xSpan = xMax - xMin || 1;
  const ySpan = yMax - yMin || 1;
  return points
    .map((p, i) => {
      const px = pad + ((p.x - xMin) / xSpan) * (w - 2 * pad);
      const py = h - pad - ((p.y - yMin) / ySpan) * (h - 2 * pad);
      return `${i === 0 ? "M" : "L"}${px.toFixed(2)},${py.toFixed(2)}`;
    })
    .join(" ");
}

export function XyChart({
  series,
  height = 220,
  title,
}: {
  series: Series[];
  height?: number;
  title?: string;
}) {
  const width = 600;
  const pad = 28;
  const allPoints = series.flatMap((s) => s.points);
  if (allPoints.length === 0) {
    return (
      <div className="flex h-[220px] items-center justify-center rounded-xl border border-dashed border-slate-700 text-sm text-slate-500">
        Aucune donnée à afficher
      </div>
    );
  }
  const xMin = Math.min(...allPoints.map((p) => p.x));
  const xMax = Math.max(...allPoints.map((p) => p.x));
  const yMin = Math.min(...allPoints.map((p) => p.y));
  const yMax = Math.max(...allPoints.map((p) => p.y));

  return (
    <div className="rounded-xl border border-slate-800 bg-slate-950/60 p-3">
      {title ? <p className="mb-2 text-xs font-medium uppercase tracking-wide text-slate-400">{title}</p> : null}
      <svg viewBox={`0 0 ${width} ${height}`} className="h-auto w-full">
        <rect x={0} y={0} width={width} height={height} fill="transparent" />
        {/* zero-line / grid */}
        <line x1={pad} y1={height - pad} x2={width - pad} y2={height - pad} stroke="#334155" strokeWidth={1} />
        <line x1={pad} y1={pad} x2={pad} y2={height - pad} stroke="#334155" strokeWidth={1} />
        {series.map((s) => (
          <path
            key={s.name}
            d={buildPath(s.points, xMin, xMax, yMin, yMax, width, height, pad)}
            fill="none"
            stroke={s.color}
            strokeWidth={2.5}
            strokeLinejoin="round"
            strokeLinecap="round"
          />
        ))}
      </svg>
      <div className="mt-2 flex flex-wrap gap-4">
        {series.map((s) => (
          <div key={s.name} className="flex items-center gap-1.5 text-xs text-slate-400">
            <span className="h-2.5 w-2.5 rounded-full" style={{ backgroundColor: s.color }} />
            {s.name}
          </div>
        ))}
      </div>
    </div>
  );
}
