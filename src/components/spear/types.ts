export interface GenerationRecord {
  generation: number;
  bestFitness: number;
  bestSize: number;
  bestExtra?: number;
}

export interface ChartPoint {
  x: number;
  target?: number;
  predicted?: number;
}

export interface SpearRunRecord {
  id: number;
  preset: string;
  label: string;
  config: Record<string, unknown>;
  status: string;
  error: string | null;
  formulaText: string | null;
  fitness: number | null;
  mse: number | null;
  linfError: number | null;
  treeSize: number | null;
  durationMs: number | null;
  history: GenerationRecord[] | null;
  metrics: Record<string, unknown> | null;
  chartData: ChartPoint[] | null;
  createdAt: string;
}
