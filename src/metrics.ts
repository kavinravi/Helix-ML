import catalog from "../local/metric_catalog.json";
import type { Metric, Task } from "./types";

const names: Partial<Record<Metric, string>> = { auroc: "ROC-AUC", f1: "F1", rmse: "RMSE", mae: "MAE", mse: "MSE", mape: "MAPE", msle: "MSLE", rmsle: "RMSLE", r2: "R²", davies_bouldin: "Davies–Bouldin", matthews_corrcoef: "MCC", calinski_harabasz: "Calinski–Harabasz" };
export const metricName = (metric: Metric) => names[metric] || metric.replaceAll("_", " ").replace(/^f1/, "F1").replace(/^roc auc/, "ROC-AUC");
export const objectiveName = (task: Pick<Task, "metric" | "metrics">) => task.metrics && task.metrics.length > 1 ? `Mean (${task.metrics.map(metricName).join(" + ")})` : metricName(task.metric);

export const metricKind = (metric: Metric) => catalog[metric].kind;
export const higherIsBetter = (metric: Metric) => catalog[metric].maximize;
