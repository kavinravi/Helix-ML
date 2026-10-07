export type AgentId = "codex" | "claude";
export type Metric = "accuracy" | "auroc" | "log_loss" | "rmse" | "mae";
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
  agent: AgentId;
  dataset: string;
  target: string;
  objective: string;
  metric: Metric;
  minutes: number;
  trials: number;
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
  taskType: "classification" | "regression";
  warnings: string[];
}
export interface Trial {
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
export interface Run {
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
