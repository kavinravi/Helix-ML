import { realpath, lstat } from "node:fs/promises";
import { resolve, relative, isAbsolute } from "node:path";
import { METRICS, CATALOG, normalizeMetric, higherIsBetter } from "./metrics.mjs";

export function validateTask(body) {
  if (!body || !["codex", "claude"].includes(body.agent))
    throw new Error("Select a signed-in agent.");
  body = { ...body, metric: normalizeMetric(body.metric), ...(Array.isArray(body.metrics) ? {metrics: body.metrics.map(normalizeMetric)} : {}) };
  const learning = body.learning ?? "supervised";
  if (!["supervised", "clustering", "reduction"].includes(learning)) throw new Error("Choose supervised learning, clustering, or dimensionality reduction.");
  const reductionMode = body.reductionMode ?? "dimensions";
  if (!["dimensions", "variance"].includes(reductionMode)) throw new Error("Choose fixed dimensions or a PCA variance target.");
  const dimensions = body.dimensions === undefined ? 2 : body.dimensions;
  const varianceTarget = body.varianceTarget === undefined ? .95 : body.varianceTarget;
  if (learning === "reduction" && reductionMode === "dimensions" && (!Number.isInteger(dimensions) || dimensions < 1 || dimensions > 50)) throw new Error("Choose between 1 and 50 output dimensions.");
  if (learning === "reduction" && reductionMode === "variance" && (!Number.isFinite(varianceTarget) || varianceTarget <= 0 || varianceTarget > 1)) throw new Error("PCA cumulative variance must be greater than 0% and at most 100%.");
  if (learning === "reduction" && reductionMode === "variance" && (body.searchModels || !["PCA", "sklearn.decomposition.PCA"].includes(body.model))) throw new Error("A cumulative variance target requires PCA.");
  const excludedColumns = body.excludedColumns ?? [];
  if (!Array.isArray(excludedColumns) || excludedColumns.length > 199 || excludedColumns.some(c => typeof c !== "string" || !c.trim() || c.length > 200) || new Set(excludedColumns.map(c => c.trim())).size !== excludedColumns.length)
    throw new Error("Excluded columns must be unique column names.");
  if (
    typeof body.dataset !== "string" ||
    !body.dataset.trim() ||
    body.dataset.length > 4096
  )
    throw new Error("Select a dataset.");
  if (
    typeof body.target !== "string" ||
    ((learning === "supervised" || CATALOG[body.metric]?.targetRequired) && !body.target.trim()) ||
    body.target.length > 200
  )
    throw new Error("Enter the target column.");
  if (
    typeof body.objective !== "string" ||
    !body.objective.trim() ||
    body.objective.length > 4000
  )
    throw new Error("Describe the prediction task.");
  if (!METRICS.includes(body.metric))
    throw new Error("Select a supported validation metric.");
  const kind=CATALOG[body.metric].kind;
  if ((learning === "supervised" ? !["classification","regression"].includes(kind) : kind!==learning)) throw new Error("Choose a metric for the selected learning mode.");
  const requested=body.metrics ?? [body.metric];
  if (!Array.isArray(requested) || !requested.length || requested.length > METRICS.length || requested[0]!==body.metric || new Set(requested).size!==requested.length || requested.some(metric=>!CATALOG[metric] || CATALOG[metric].kind!==kind)) throw new Error("Choose unique metrics for the same task; the first must match the main metric.");
  if (requested.length>1 && (learning!=="supervised" || requested.some(metric=>higherIsBetter(metric)!==higherIsBetter(body.metric)))) throw new Error("Joint optimization needs metrics with the same direction. Choose a primary objective when combining a score to maximize with a loss to minimize.");
  if (body.positiveClass!==undefined && (typeof body.positiveClass!=="string" || !body.positiveClass.trim() || body.positiveClass.length>200 || kind!=="classification")) throw new Error("A positive class must be a label for a classification objective.");
  if (
    body.minutes !== null && (!Number.isInteger(body.minutes) ||
    body.minutes < 1 ||
    body.minutes > 1440)
  )
    throw new Error("The time budget must be between 1 and 1440 minutes.");
  if (body.trials !== null && (!Number.isInteger(body.trials) || body.trials < 1 || body.trials > 100))
    throw new Error("The trial budget must be between 1 and 100.");
  if (body.minutes === null && body.trials === null)
    throw new Error("Enable at least one run limit: max time or max trials.");
  if (
    typeof body.searchModels !== "boolean" ||
    typeof body.model !== "string" ||
    body.model.length > 2000 ||
    (!body.searchModels && !body.model.trim())
  )
    throw new Error(
      "Specify a model or model family when model search is disabled.",
    );
  if (!["py", "ipynb"].includes(body.output))
    throw new Error("Choose Python or notebook output.");
  if (
    typeof body.exportModel !== "boolean" ||
    ![
      "native",
      "joblib",
      "pickle",
      "pytorch",
      "torchscript",
      "keras",
      "savedmodel",
      "onnx",
    ].includes(body.exportFormat)
  )
    throw new Error("Choose a supported model export format.");
  if (
    !["holdout", "cv"].includes(body.validation) ||
    ![3, 5, 10].includes(body.folds)
  )
    throw new Error("Choose a validation method and 3, 5, or 10 folds.");
  const testFraction = body.testFraction ?? 0.2;
  const holdoutFraction = body.holdoutFraction ?? 0.2;
  if (!Number.isFinite(testFraction) || testFraction < 0 || testFraction >= 1)
    throw new Error("The test split must be at least 0% and less than 100%.");
  if (!Number.isFinite(holdoutFraction) || holdoutFraction <= 0 || holdoutFraction >= 1)
    throw new Error("The validation split must be greater than 0% and less than 100% of the training pool.");
  if (
    !Array.isArray(body.seeds) ||
    !body.seeds.length ||
    body.seeds.length > 5 ||
    body.seeds.some(
      (seed) => !Number.isInteger(seed) || seed < 0 || seed > 2 ** 32 - 1,
    ) ||
    new Set(body.seeds).size !== body.seeds.length
  )
    throw new Error("Provide up to five unique nonnegative 32-bit seeds.");
  const keys = [
    "augmentation",
    "regularization",
    "features",
    "tuning",
    "pretrained",
    "ensemble",
  ];
  if (!body.policy || keys.some((key) => typeof body.policy[key] !== "boolean"))
    throw new Error("Set all six model permissions.");
  if (learning !== "supervised" && body.policy.ensemble) throw new Error("Ensembling is currently available for supervised learning only.");
  const splitStrategy = body.splitStrategy ?? "independent";
  const groupColumn = body.groupColumn ?? "";
  const timeColumn = body.timeColumn ?? "";
  const assetColumns = body.assetColumns ?? [];
  if (!["independent", "group", "time"].includes(splitStrategy))
    throw new Error("Choose independent rows, groups, or chronological evaluation.");
  if ([groupColumn, timeColumn].some((c) => typeof c !== "string" || c.length > 200))
    throw new Error("Group and time columns must be column names.");
  if ((splitStrategy === "group" && !groupColumn.trim()) || (splitStrategy === "time" && !timeColumn.trim()))
    throw new Error("Enter the column used to separate groups or order timestamps.");
  if (!Array.isArray(assetColumns) || assetColumns.length > 20 || assetColumns.some((c) => typeof c !== "string" || !c.trim() || c.length > 200) || new Set(assetColumns.map((c) => c.trim())).size !== assetColumns.length)
    throw new Error("Provide up to 20 unique asset column names.");
  return {
    learning, dimensions, reductionMode, varianceTarget, excludedColumns: excludedColumns.map(c => c.trim()),
    agent: body.agent,
    dataset: body.dataset.trim(),
    target: learning === "supervised" || CATALOG[body.metric].targetRequired ? body.target.trim() : "",
    objective: body.objective.trim(),
    metric: body.metric,
    ...(requested.length > 1 ? { metrics: [...requested] } : {}),
    ...(body.positiveClass ? { positiveClass: body.positiveClass.trim() } : {}),
    minutes: body.minutes,
    trials: body.trials,
    searchModels: body.searchModels,
    model: body.model.trim(),
    output: body.output,
    exportModel: body.exportModel,
    exportFormat: body.exportFormat,
    validation: body.validation,
    folds: body.folds,
    testFraction,
    holdoutFraction,
    seeds: [...body.seeds],
    splitStrategy,
    groupColumn: groupColumn.trim(),
    timeColumn: timeColumn.trim(),
    assetColumns: assetColumns.map((c) => c.trim()),
    policy: Object.fromEntries(keys.map((key) => [key, body.policy[key]])),
  };
}

export async function inside(root, name) {
  if (
    typeof name !== "string" ||
    !name ||
    isAbsolute(name) ||
    name.includes("\0") ||
    name.includes("\\") ||
    name.split("/").some((part) => part === "." || part === "..")
  )
    throw new Error("Invalid file path.");
  const absolute = resolve(root, name);
  const rel = relative(root, absolute);
  if (rel.startsWith("..") || isAbsolute(rel))
    throw new Error("The file is outside this workspace.");
  const actual = await realpath(absolute);
  const actualRel = relative(await realpath(root), actual);
  if (
    actualRel.startsWith("..") ||
    isAbsolute(actualRel) ||
    !(await lstat(actual)).isFile()
  )
    throw new Error("The file is outside this workspace.");
  return actual;
}

export const redact = (text) =>
  String(text)
    .replace(
      /(?:sk-[A-Za-z0-9_-]{12,}|Bearer\s+[A-Za-z0-9_.-]{20,})/g,
      "[redacted]",
    )
    .slice(0, 6000);
