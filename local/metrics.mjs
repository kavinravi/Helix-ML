export const METRICS = ["accuracy", "auroc", "log_loss", "rmse", "mae", "silhouette", "davies_bouldin", "trustworthiness"];
export const higherIsBetter = (metric) =>
  ["accuracy", "auroc", "silhouette", "trustworthiness"].includes(metric);
export const isBetter = (metric, score, best) =>
  best === null || (higherIsBetter(metric) ? score > best : score < best);

export function scorePredictions(metric, truth, predictions, classes = []) {
  if (!["accuracy", "auroc", "log_loss", "rmse", "mae"].includes(metric)) throw new Error("Unsupported prediction metric");
  if (!Array.isArray(truth) || !truth.length)
    throw new Error("Evaluation targets must be a nonempty array.");
  if (["accuracy", "auroc", "log_loss"].includes(metric)) {
    if (classes.length && (new Set(classes.map(String)).size !== classes.length || truth.some((v) => !classes.map(String).includes(String(v)))))
      throw new Error("Unknown or duplicate evaluation class.");
    if (metric !== "accuracy" && classes.length < 2)
      throw new Error("Probability metrics require at least two known classes.");
  } else if (truth.some((v) => !["number", "string"].includes(typeof v) || (typeof v === "string" && !v.trim()) || !Number.isFinite(Number(v)))) {
    throw new Error("Regression targets must be finite numbers.");
  }
  if (
    !Array.isArray(predictions) ||
    predictions.length !== truth.length ||
    truth.length === 0
  )
    throw new Error(
      `Expected ${truth.length} predictions, one for each validation row.`,
    );
  const n = truth.length;
  if (metric === "accuracy") {
    if (
      predictions.some(
        (value) => !["string", "number", "boolean"].includes(typeof value) || (typeof value === "number" && !Number.isFinite(value)) || (classes.length && !classes.map(String).includes(String(value))),
      )
    )
      throw new Error("Accuracy requires class labels, not probabilities.");
    return (
      predictions.reduce(
        (sum, value, i) => sum + (String(value) === String(truth[i]) ? 1 : 0),
        0,
      ) / n
    );
  }
  if (metric === "rmse" || metric === "mae") {
    if (
      predictions.some(
        (value) => typeof value !== "number" || !Number.isFinite(value),
      )
    )
      throw new Error("Regression predictions must be finite numbers.");
    const error =
      predictions.reduce(
        (sum, value, i) =>
          sum +
          (metric === "rmse"
            ? (value - Number(truth[i])) ** 2
            : Math.abs(value - Number(truth[i]))),
        0,
      ) / n;
    if (!Number.isFinite(error)) throw new Error("Regression error overflowed; rescale the target values.");
    return metric === "rmse" ? Math.sqrt(error) : error;
  }
  if (metric === "auroc") {
    if (classes.length !== 2)
      throw new Error("AUROC requires exactly two classes.");
    if (
      predictions.some(
        (value) =>
          typeof value !== "number" ||
          !Number.isFinite(value) ||
          value < 0 ||
          value > 1,
      )
    )
      throw new Error(
        "AUROC requires probabilities between 0 and 1 for classes[1].",
      );
    const sorted = truth
      .map((label, i) => ({
        positive: String(label) === String(classes[1]),
        p: predictions[i],
      }))
      .sort((a, b) => a.p - b.p);
    const positive = sorted.filter((row) => row.positive).length;
    if (!positive || positive === n)
      throw new Error("Validation needs both classes to compute AUROC.");
    let ranks = 0;
    for (let i = 0; i < n; ) {
      let end = i + 1;
      while (end < n && sorted[end].p === sorted[i].p) end++;
      const rank = (i + 1 + end) / 2;
      for (let j = i; j < end; j++) if (sorted[j].positive) ranks += rank;
      i = end;
    }
    return (
      (ranks - (positive * (positive + 1)) / 2) / (positive * (n - positive))
    );
  }
  let total = 0;
  for (let i = 0; i < n; i++) {
    const row = predictions[i];
    if (
      !Array.isArray(row) ||
      row.length !== classes.length ||
      row.some(
        (p) => typeof p !== "number" || !Number.isFinite(p) || p < 0 || p > 1,
      ) ||
      Math.abs(row.reduce((a, b) => a + b, 0) - 1) > 0.001
    )
      throw new Error(
        "Log loss requires probability arrays in classes.json order, each summing to 1.",
      );
    const index = classes.findIndex(
      (label) => String(label) === String(truth[i]),
    );
    if (index < 0) throw new Error("Unknown validation class");
    total -= Math.log(Math.min(1 - Number.EPSILON, Math.max(Number.EPSILON, row[index])));
  }
  return total / n;
}
