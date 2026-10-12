import { mkdir, readFile, writeFile, cp } from "node:fs/promises";
import { join } from "node:path";
import { randomInt, createHash } from "node:crypto";
import { container } from "../local/runtime.mjs";
import { callTool } from "../local/tools.mjs";
import { harness, dependencyInstallCommand } from "../local/export.mjs";
import { redact } from "../local/validate.mjs";
import { runProcess, remainingTime } from "../local/process.mjs";
import { log, saveRun } from "../local/engine.mjs";
import { TORCH_VERSION } from "./compute.mjs";

export const RL_VERSIONS = { "stable-baselines3": "2.9.0", gymnasium: "1.4.0" };
export const RL_ENVIRONMENTS = {
  "CartPole-v1": "classic-control", "MountainCar-v0": "classic-control", "MountainCarContinuous-v0": "classic-control",
  "Pendulum-v1": "classic-control", "Acrobot-v1": "classic-control",
  "FrozenLake-v1": "toy-text", "Taxi-v3": "toy-text", "CliffWalking-v1": "toy-text", "Blackjack-v1": "toy-text",
  "LunarLander-v3": "box2d", "BipedalWalker-v3": "box2d", "CarRacing-v3": "box2d",
  ...Object.fromEntries(["Ant", "HalfCheetah", "Hopper", "Humanoid", "HumanoidStandup", "InvertedDoublePendulum", "InvertedPendulum", "Pusher", "Reacher", "Swimmer", "Walker2d"].map(name => [name + "-v5", "mujoco"])),
};
export function validateRlBudget(task) {
  for (const [key, max] of Object.entries({minutes:1440, trials:100}))
    if (task[key] !== null && (!Number.isInteger(task[key]) || task[key] < 1 || task[key] > max)) throw new Error(`${key} must be an integer between 1 and ${max}, or null.`);
  if (task.minutes === null && task.trials === null) throw new Error("Keep at least one time or trial budget.");
  return task;
}
export function rlTask(args) {
  if (!Object.hasOwn(RL_ENVIRONMENTS, args.envId)) throw new Error("Choose a supported standard Gymnasium environment from helix_info.");
  if (!args.objective.trim() || args.objective.length > 4000) throw new Error("Provide an objective under 4,000 characters.");
  return validateRlBudget({ ...args, learning: "reinforcement", metric: "episode_return", minutes: args.minutes === undefined ? 30 : args.minutes,
    trials: args.trials === undefined ? 12 : args.trials, seed: args.seed ?? 42, evaluationEpisodes: args.evaluationEpisodes ?? 10,
    maxEpisodeSteps: args.maxEpisodeSteps ?? null, policy: { pretrained: false }, searchModels: true });
}
export function rlContract(run) {
  return `Standard Gymnasium ${run.task.envId}. Use helix_run_rl_trial with an SB3 algorithm, totalTimesteps and optional hyperparameters. Each trial trains a fresh policy using seed ${run.task.seed}. Rank by mean undiscounted episode return on the same ${run.task.evaluationEpisodes} validation seeds. Training/evaluation episode limits match. Evaluation is deterministic-policy inference in a fresh process; rewards are measured by the environment. Finalize the selected checkpoint once on separate reserved episode seeds. SB3 rollout batches may exceed requested timesteps; actual timesteps are recorded. Small MLP policies generally benefit from CPU; explicitly request CUDA/MPS only when appropriate. No CSV, target column or hand-written train.py is required. No custom environments, reward wrappers or pretrained policies in this workflow.`;
}
const read = async path => JSON.parse(await readFile(path, "utf8"));
const write = (path, value) => writeFile(path, JSON.stringify(value, null, 2));
function paths(root, run) {
  const folder = join(root, run.id);
  return { folder, packages: join(folder, "dependencies", "rl") };
}
async function python(context, packages, command, signal, output, onLine) {
  return container(context, ["python", "/harness/rl.py", ...command], { signal, writable: output,
    mounts: [[harness, "/harness"], [packages, "/packages"]], onLine });
}
export async function prepareRl(root, run, context, signal) {
  const { folder, packages } = paths(root, run);
  await mkdir(packages, { recursive: true });
  const dependencies = [];
  if (!context.runtime.compute?.torchVersion) dependencies.push({ name: "torch", version: TORCH_VERSION });
  dependencies.push({ name: "stable-baselines3", version: RL_VERSIONS["stable-baselines3"] });
  const group = RL_ENVIRONMENTS[run.task.envId];
  dependencies.push({ name: group === "toy-text" ? "gymnasium" : `gymnasium[${group}]`, version: RL_VERSIONS.gymnasium });
  for (const pkg of dependencies) {
    log(run, "system", `Setting up ${pkg.name} ${pkg.version}. First use may download framework packages.`);
    await callTool("install_package", pkg, { ...context, signal, packages });
  }
  await write(join(folder, "rl-task.json"), run.task);
  await python(context, packages, ["check", "--config", join(folder, "rl-task.json"), "--output", join(folder, "environment.json")], signal);
  const seeds = new Set([run.task.seed]);
  while (seeds.size < 1 + 2 * run.task.evaluationEpisodes) seeds.add(randomInt(0, 2 ** 31));
  const values = [...seeds].slice(1);
  const protocol = { environment: await read(join(folder, "environment.json")), trainingSeed: run.task.seed,
    validationSeeds: values.slice(0, run.task.evaluationEpisodes), testSeeds: values.slice(run.task.evaluationEpisodes), metric: "mean undiscounted episode return", deterministic: true };
  run.protocolHash = createHash("sha256").update(JSON.stringify(protocol)).digest("hex");
  await write(join(folder, "rl-protocol.json"), protocol);
  run.rlEnvironment = protocol.environment;
  run.prepared = true;
  log(run, "system", "Environment ready. Propose an algorithm and training timesteps.");
}
export async function runRlTrial(root, run, args, context, signal) {
  const { folder, packages } = paths(root, run), id = `trial-${run.trials.length + 1}`;
  const trial = { id, status: "running", createdAt: new Date().toISOString(), score: null, device: context.runtime.device,
    algorithm: args.algorithm, requestedTimesteps: args.totalTimesteps, hyperparameters: args.hyperparameters || {} };
  run.trials.push(trial);
  const output = join(folder, "trials", id);
  await mkdir(output, { recursive: true });
  try {
    const config = { ...run.task, ...trial, totalTimesteps: args.totalTimesteps, maxEpisodeSteps: run.rlEnvironment.maxEpisodeSteps };
    await write(join(output, "config.json"), config);
    log(run, "system", `Training ${args.algorithm} on ${run.task.envId} (${trial.device.toUpperCase()}).`);
    await python(context, packages, ["train", "--config", join(output, "config.json"), "--output", output], signal, output, line => log(run, "system", line));
    const protocol = await read(join(folder, "rl-protocol.json"));
    await write(join(output, "evaluation.json"), { ...config, seeds: protocol.validationSeeds });
    await python(context, packages, ["evaluate", "--config", join(output, "evaluation.json"), "--model", join(output, "policy.zip"), "--output", join(output, "validation.json")], signal, output);
    const result = await read(join(output, "validation.json"));
    Object.assign(trial, result, { status: "completed", training: await read(join(output, "training.json")) });
    if (run.score === null || result.score > run.score) { run.best = id; run.score = result.score; }
    log(run, "system", `Validation episode return: ${result.score} ± ${result.deviation}.`);
  } catch (error) { trial.status = signal.aborted ? "cancelled" : "failed"; trial.error = redact(error.message); throw error; }
}
export async function finalizeRl(root, run, context, signal) {
  const { folder, packages } = paths(root, run), final = join(folder, "final"), selected = join(folder, "trials", run.best);
  run.selectionFrozen = true;
  await saveRun(root, run);
  await mkdir(final, { recursive: true });
  const protocol = await read(join(folder, "rl-protocol.json"));
  const config = await read(join(selected, "config.json"));
  if (run.rlTestStarted && !run.rlTestResult) throw new Error("The final evaluation was interrupted. Reserved test episodes cannot be run again in this experiment.");
  if (!run.rlTestStarted) {
    await cp(join(selected, "policy.zip"), join(final, "policy.zip"));
    await write(join(final, "test-config.json"), { ...config, seeds: protocol.testSeeds });
    run.rlTestStarted = true;
    await saveRun(root, run);
    await python(context, packages, ["evaluate", "--config", join(final, "test-config.json"), "--model", join(final, "policy.zip"), "--output", join(final, "test.json")], signal, final);
    run.rlTestResult = await read(join(final, "test.json"));
    run.testScore = run.rlTestResult.score;
    await saveRun(root, run);
  }
  await cp(join(harness, "rl.py"), join(final, "rl.py"));
  await cp(join(selected, "config.json"), join(final, "config.json"));
  await cp(join(selected, "training.json"), join(final, "training.json"));
  await write(join(final, "protocol.json"), { ...protocol, sha256: run.protocolHash });
  const versions = await container(context, ["python", "-c", "import importlib.metadata as m; d={}; [d.setdefault(x.metadata['Name'].lower(),x.version) for x in m.distributions() if x.metadata['Name']]; print('\\n'.join(k+'=='+v for k,v in sorted(d.items())))"], { signal, mounts: [[packages, "/packages"]] });
  await writeFile(join(final, "requirements.lock"), versions.output);
  await write(join(final, "experiment.json"), { task: run.task, environment: run.environment, trials: run.trials, selected: run.best, validationScore: run.score, test: run.rlTestResult, protocolHash: run.protocolHash });
  await writeFile(join(final, "README.md"), `# Helix reinforcement learning experiment\n\n${run.task.envId}, ${config.algorithm}. Validation mean return: ${run.score}; reserved test mean return: ${run.testScore}.\n\nThe selected checkpoint was reloaded in a fresh process for each evaluation. No final retraining or selection on test rewards. Episode returns are undiscounted, policies deterministic, and episode horizons match training. Results describe one training seed, not performance across independently trained seeds.\n\nUse Python 3.12 and run ${dependencyInstallCommand(versions.output)}. The environment and exact dependency versions must be compatible with your platform. Load only checkpoints you trust.\n\nEvaluate the exported policy:\n\n\`\`\`sh\npython rl.py evaluate --config test-config.json --model policy.zip --output reproduced-test.json\n\`\`\`\n\nSet HELIX_DEVICE=cpu for CPU inference. To train a new policy with the recorded parameters: python rl.py train --config config.json --output new-training\n\nprotocol.json records separate training, validation and test seeds. Training timesteps can exceed the requested count to finish an SB3 rollout batch. training.json records actual timesteps and measured episode rewards.\n`);
  await runProcess(context.runtime.python, [join(harness, "bundle.py"), final], { signal, timeout: remainingTime(context.deadline) });
  run.bundle = join(final, "helix-solution.zip");
}
