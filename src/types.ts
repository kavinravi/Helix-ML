export type AgentId = "codex" | "claude";
export type Metric = keyof typeof import("../local/metric_catalog.json");
export type ExportFormat =
  | "native"
  | "joblib"
  | "pickle"
  | "pytorch"
  | "torchscript"
  | "keras"
  | "savedmodel"
  | "onnx";
export type RunStatus =
  | "queued"
  | "running"
  | "paused"
  | "completed"
  | "failed"
  | "stopped";
export interface Provider {
  id: AgentId;
  name: string;
  installed: boolean;
  authenticated: boolean;
  detail: string;
  version?: string;
  capability?: AgentCapability;
}
export interface AgentCapability {
  status: "unchecked" | "checking" | "verified" | "failed";
  detail: string;
  verifiedAt?: string;
  checks?: string[];
}
export interface Connection {
  url: string;
  token: string;
  desktop?: boolean;
}
export interface Policy {
  augmentation: boolean;
  regularization: boolean;
  features: boolean;
  tuning: boolean;
  pretrained: boolean;
  ensemble: boolean;
}
export interface Task {
  learning?: "supervised" | "clustering" | "reduction";
  dimensions?: number;
  reductionMode?: "dimensions" | "variance";
  varianceTarget?: number;
  excludedColumns?: string[];
  agent: AgentId;
  dataset: string;
  target: string;
  objective: string;
  metric: Metric;
  metrics?: Metric[];
  positiveClass?: string;
  minutes: number | null;
  trials: number | null;
  searchModels: boolean;
  model: string;
  output: "py" | "ipynb";
  exportModel: boolean;
  exportFormat: ExportFormat;
  validation: "holdout" | "cv";
  folds: number;
  testFraction: number;
  holdoutFraction: number;
  seeds: number[];
  splitStrategy: "independent" | "group" | "time";
  groupColumn: string;
  timeColumn: string;
  assetColumns: string[];
  policy: Policy;
}
export interface DatasetInfo {
  fingerprint: string;
  rows: number;
  developmentRows: number;
  testRows: number;
  fitsPerTrial: number;
  assetBytes: number;
  assets: Record<string, { bytes: number; sha256: string }>;
  schema: { name: string; type: string; missing: number }[];
  taskType: "classification" | "regression" | "clustering" | "reduction";
  warnings: string[];
}
export interface Trial {
  metricScores?: Partial<Record<Metric, number>>;
  id: string;
  phase: string;
  name: string;
  score: number | null;
  status: "accepted" | "rejected" | "failed";
  duration: number;
  detail: string;
  artifact: string;
  component?: string;
  deviation?: number;
  interval?: number[];
  foldScores?: number[];
}
export interface Log {
  time: string;
  type: string;
  message: string;
}
export interface FollowupMessage {
  id: string;
  message: string;
  mode: "chat" | "trials";
  status: "pending" | "completed" | "failed";
  createdAt: string;
  reply?: string;
  error?: string;
  childRunId?: string;
}
export interface Run {
  metricScores?: Partial<Record<Metric, number>>;
  testMetricScores?: Partial<Record<Metric, number>>;
  projection?: { kind: "clustering" | "reduction"; axes: string[]; rows: number; scope: string; dimensions?: number; cumulativeVariance?: number; clusters?: { label: number; count: number }[]; points: { row: number; x: number; y: number; cluster?: number }[] };
  messages?: FollowupMessage[];
  followup?: { parentId: string; message: string; protocolHash: string };
  trainingHistory?: { step: number; loss?: number; accuracy?: number }[];
  id: string;
  task: Task;
  status: RunStatus;
  phase: string;
  createdAt: string;
  startedAt: string | null;
  elapsed: number;
  trials: Trial[];
  logs: Log[];
  best: string | null;
  score: number | null;
  testScore?: number | null;
  baseline?: { name: string; score: number; deviation: number } | null;
  testBaseline?: { name: string; score: number; deviation: number } | null;
  next: number;
  progress?: { trial: number; stage: string; completedFits: number; totalFits: number };
  error?: string;
}
export interface Tool {
  id: string;
  name: string;
  description: string;
  status: string;
  kind: string;
}
