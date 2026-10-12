import { CATALOG } from "../local/metrics.mjs";
import { tools as internalTools } from "../local/tools.mjs";
import { RL_ENVIRONMENTS } from "./rl.mjs";

const string = { type: "string" }, integer = { type: "integer" }, boolean = { type: "boolean" };
const budget = { type: ["integer", "null"], minimum: 1 };
const object = (properties, required = []) => ({ type: "object", properties, required, additionalProperties: false });
const id = { runId: string };
const tool = (name, description, properties, required = [], readOnly = false) => ({ name: "helix_" + name, description, inputSchema: object(properties, required), annotations: { readOnlyHint: readOnly, destructiveHint: !readOnly, openWorldHint: !readOnly, idempotentHint: readOnly } });

export const tools = [
  tool("info", "Read local runtime, supported metrics and workflow. No account or API key is needed; the calling agent drives experiments.", {}, [], true),
  tool("create_rl_experiment", "Prepare a standard Gymnasium reinforcement learning study. No CSV is needed. Show environment, episode horizon, evaluation episode count, time/trial budgets and device before calling. Packages install automatically on first use. CPU is the default for small policies. Validation and final test use separate fixed episode seeds. Returns immediately; poll status.", {
    envId: { ...string, enum: Object.keys(RL_ENVIRONMENTS) }, objective: string,
    device: { ...string, enum: ["cpu", "cuda", "mps"] }, minutes: budget, trials: budget,
    seed: { ...integer, minimum: 0, maximum: 2147483647 }, evaluationEpisodes: { ...integer, minimum: 2, maximum: 1000 },
    maxEpisodeSteps: { ...integer, minimum: 1, maximum: 100000 },
  }, ["envId", "objective"]),
  tool("run_rl_trial", "Train a fresh Stable-Baselines3 policy, save it, and measure validation episode rewards after reloading in a separate process. No source directory is required. Algorithms must support the environment's action space. Each call consumes one trial. Timesteps may round up to finish an SB3 rollout batch. Default SB3 hyperparameters apply when omitted.", {
    ...id, algorithm: { ...string, enum: ["PPO", "A2C", "DQN", "SAC", "TD3", "DDPG"] },
    totalTimesteps: { ...integer, minimum: 1, maximum: 1000000000 },
    hyperparameters: object({
      learning_rate: { type: "number", minimum: 0.000000001, maximum: 1 }, gamma: { type: "number", minimum: 0, maximum: 1 },
      batch_size: { ...integer, minimum: 2, maximum: 1000000 }, n_steps: { ...integer, minimum: 2, maximum: 1000000 },
      n_epochs: { ...integer, minimum: 1, maximum: 1000 }, buffer_size: { ...integer, minimum: 1, maximum: 10000000 },
      learning_starts: { ...integer, minimum: 0, maximum: 1000000000 }, train_freq: { ...integer, minimum: 1, maximum: 1000000 },
      gradient_steps: { ...integer, minimum: 1, maximum: 1000000 }, tau: { type: "number", minimum: 0, maximum: 1 },
      ent_coef: { type: "number", minimum: 0, maximum: 100 }, clip_range: { type: "number", minimum: 0, maximum: 1 },
      net_arch: { type: "array", maxItems: 10, items: { ...integer, minimum: 1, maximum: 8192 } },
    }),
  }, ["runId", "algorithm", "totalTimesteps"]),
  tool("create_experiment", "Prepare a local CSV experiment with immutable validation splits. Show the user the proposed task, target, metrics, budgets and restrictions before calling. Returns immediately; poll helix_status. Default: 30 minutes, 12 trials, 20% test, 20% validation of the remaining pool. Minutes measure wall time from creation, including agent work. Use null to disable one budget.", {
    dataset: { ...string, description: "Absolute path to the user's local CSV. Assets resolve relative to it." },
    objective: string, target: string, metric: { ...string, enum: Object.keys(CATALOG) }, metrics: { type: "array", items: string }, positiveClass: string,
    learning: { ...string, enum: ["supervised", "clustering", "reduction"] },
    model: { ...string, description: "Optional model/family restriction; empty permits model search." },
    device: { ...string, enum: ["auto", "cpu", "cuda", "mps"], description: "Default auto prefers a verified GPU. Explicit cpu/cuda/mps requires every candidate to use that device. Auto permits CPU candidates for CPU-only algorithms." },
    minutes: budget, trials: budget, validation: { ...string, enum: ["holdout", "cv"] }, folds: { ...integer, enum: [3, 5, 10] },
    testFraction: { type: "number" }, holdoutFraction: { type: "number" }, seeds: { type: "array", items: integer },
    excludedColumns: { type: "array", items: string }, assetColumns: { type: "array", items: string },
    splitStrategy: { ...string, enum: ["independent", "group", "time"] }, groupColumn: string, timeColumn: string,
    dimensions: integer, reductionMode: { ...string, enum: ["dimensions", "variance"] }, varianceTarget: { type: "number" },
    policy: object(Object.fromEntries(["augmentation", "regularization", "features", "tuning", "pretrained", "ensemble"].map(key => [key, boolean]))),
    output: { ...string, enum: ["py", "ipynb"] }, exportModel: boolean,
    exportFormat: { ...string, enum: ["native", "joblib", "pickle", "pytorch", "torchscript", "keras", "savedmodel", "onnx"] },
  }, ["dataset", "objective", "metric"]),
  tool("status", "Read compact progress, recent measured trials and artifact paths. Request details=true once preparation finishes for the dataset schema, training contract and all trial results. Wait several seconds between polls. No test scores are exposed before finalization.", { ...id, details: boolean }, ["runId"], true),
  tool("list_experiments", "List saved local experiments across sessions, newest first.", {}, [], true),
  tool("run_trial", "Snapshot reviewed Python source and run one candidate on all fixed folds/seeds. Uses this computer with normal user permissions. Returns immediately; poll status. Include train.py and plan.json (component, change, rationale, modelFamily, strategies). For export include predict.py. Only submit source code, never credentials/data. Failed submissions after job start consume one trial. Optional packages are installed only for this candidate; reuse the base environment when possible.", {
    ...id, sourceDir: { ...string, description: "Absolute directory with candidate Python source and plan.json, outside Helix's state folder." },
    device: { ...string, enum: ["cpu", "cuda", "mps"], description: "Candidate's actual training device. Defaults to the experiment preference. Choose cpu for sklearn/CPU-only models in auto experiments. Must honor an explicit experiment device." },
    packages: { type: "array", maxItems: 20, items: { ...internalTools.find(t => t.name === "install_package").inputSchema, required: ["name", "version"] } },
  }, ["runId", "sourceDir"]),
  tool("finalize", "Freeze the best validation candidate and evaluate the reserved test partition ONCE, verify model reload and export a solution ZIP. CSV models are refitted; RL evaluates the selected saved policy on reserved episode seeds without retraining. No further trials may be submitted to this run. Reserve time for finalization; extend the budget beforehand if necessary. Returns immediately; poll status.", id, ["runId"]),
  tool("cache_model", "Download a public pretrained model at a pinned revision when pretrained models are permitted. Does not download remote Python or pickle weights. Returns immediately; poll status. Training should resolve the returned cache directory under its --models path.", { ...id, ...internalTools.find(t => t.name === "cache_model").inputSchema.properties }, ["runId", "repository", "revision"]),
  tool("cancel", "Stop the active job and its child processes. Completed validation trials are preserved. An interrupted final test cannot be repeated.", id, ["runId"]),
  tool("extend_budget", "Set NEW total minutes/trials before finalization, when the user requests further work. Minutes are measured from original creation, so include elapsed wall time. Null disables one limit; at least one must remain. Never change data, metrics or validation splits.", { ...id, minutes: budget, trials: budget }, ["runId", "minutes", "trials"]),
];

// Reject unknown keys/types even when a client bypasses its own schema validation.
export function validateArguments(schema, value, path = "arguments") {
  const type = value === null ? "null" : Array.isArray(value) ? "array" : typeof value;
  const types = Array.isArray(schema.type) ? schema.type : [schema.type];
  if (!types.includes(type) && !(types.includes("integer") && Number.isInteger(value))) throw new Error(`${path} has an invalid type.`);
  if (schema.enum && !schema.enum.includes(value)) throw new Error(`${path} is not a supported choice.`);
  if (typeof value === "number" && (!Number.isFinite(value) || (types.includes("integer") && !Number.isSafeInteger(value)) || (schema.minimum !== undefined && value < schema.minimum) || (schema.maximum !== undefined && value > schema.maximum))) throw new Error(`${path} is out of range.`);
  if (type === "object") {
    for (const key of schema.required || []) if (!(key in value)) throw new Error(`${path}.${key} is required.`);
    for (const key of Object.keys(value)) {
      if (!Object.hasOwn(schema.properties, key)) throw new Error(`Unknown option ${path}.${key}.`);
      validateArguments(schema.properties[key], value[key], `${path}.${key}`);
    }
  }
  if (type === "array") {
    if (value.length > (schema.maxItems ?? 200)) throw new Error(`${path} has too many items.`);
    value.forEach((item, index) => validateArguments(schema.items, item, `${path}[${index}]`));
  }
}
