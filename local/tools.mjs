import { readFile, readdir, mkdir, writeFile, lstat } from "node:fs/promises";
import { join } from "node:path";
import { runProcess, expired } from "./process.mjs";
import { container, IMAGE } from "./runtime.mjs";
import { inside } from "./validate.mjs";

export { IMAGE };
const packages = [
  "numpy",
  "pandas",
  "scikit-learn",
  "umap-learn",
  "scipy",
  "pillow",
  "torch",
  "torchvision",
  "torchaudio",
  "timm",
  "transformers",
  "datasets",
  "accelerate",
  "sentence-transformers",
  "lightgbm",
  "xgboost",
  "catboost",
  "librosa",
  "soundfile",
  "opencv-python-headless",
  "huggingface-hub",
  "safetensors",
  "joblib",
  "nbformat",
  "nbclient",
  "ipykernel",
  "onnx",
  "onnxruntime",
  "skl2onnx",
  "tensorflow-cpu",
];
const objectSchema = (properties) => ({
  type: "object",
  properties,
  additionalProperties: false,
});
export const tools = [
  {
    name: "read_source",
    description: "Read a candidate source, plan, audit, requirements or README file inside the current workspace, including ensemble members. Use this to inspect code before reviewing or editing it.",
    inputSchema: objectSchema({ name: { type: "string" } }),
  },
  {
    name: "write_source",
    description: "Write a Python source file or candidate/plan/audit JSON in the current candidate directory. Does not execute it.",
    inputSchema: objectSchema({ name: { type: "string" }, content: { type: "string" } }),
  },
  {
    name: "dataset_info",
    description:
      "Read the fixed training/validation manifest and class order. Validation targets are excluded. Unsupervised tasks have no target; their output constraints are included.",
    inputSchema: objectSchema({}),
  },
  {
    name: "previous_experiments",
    description: "Read actual scores and plans recorded by the runner.",
    inputSchema: objectSchema({}),
  },
  {
    name: "search_models",
    description: "Search Hugging Face for task-appropriate pretrained models.",
    inputSchema: objectSchema({
      query: { type: "string" },
      task: { type: "string" },
    }),
  },
  {
    name: "search_papers",
    description:
      "Find ML research papers on arXiv. Search methods, not competition solutions.",
    inputSchema: objectSchema({ query: { type: "string" } }),
  },
  {
    name: "github_file",
    description:
      "Read a source file from a public GitHub repository, using the local gh login when available.",
    inputSchema: objectSchema({
      repository: { type: "string" },
      path: { type: "string" },
      ref: { type: "string" },
    }),
  },
  {
    name: "kaggle_files",
    description:
      "List competition data files through the locally authenticated Kaggle CLI.",
    inputSchema: objectSchema({ competition: { type: "string" } }),
  },
  {
    name: "install_package",
    description:
      "Install an approved ML dependency into this candidate's isolated directory. First read dataset_info.installedPackages; use the existing runtime version when compatible. Install only a missing or incompatible dependency. Changes apply only to this candidate and its descendants.",
    inputSchema: objectSchema({ name: { type: "string", enum: packages }, version: { type: "string" } }),
  },
  {
    name: "cache_model",
    description:
      "Download a Hugging Face model into this run’s model cache. Training has no network access.",
    inputSchema: objectSchema({ repository: { type: "string" }, revision: { type: "string" } }),
  },
  {
    name: "list_artifacts",
    description: "List the current candidate’s files.",
    inputSchema: objectSchema({}),
  },
];

export function availableTools(task, mode) {
  return tools.filter((tool) =>
    (mode !== "setup" || ["read_source", "write_source"].includes(tool.name)) &&
    (mode !== "discussion" || ["read_source", "dataset_info", "previous_experiments", "list_artifacts"].includes(tool.name)) &&
    (tool.name !== "search_models" || (task.searchModels && task.policy.pretrained)) &&
    (tool.name !== "cache_model" || task.policy.pretrained)).map((tool) => ({
      ...tool,
      annotations: { readOnlyHint: !["write_source", "install_package", "cache_model"].includes(tool.name), destructiveHint: false,
        openWorldHint: ["search_models", "search_papers", "github_file", "kaggle_files", "install_package", "cache_model"].includes(tool.name) },
    }));
}

async function fetchJSON(url) {
  const response = await fetch(url, {
    signal: AbortSignal.timeout(20_000),
    headers: { "User-Agent": "helix-ml/0.1" },
  });
  if (!response.ok) throw new Error(`Source returned HTTP ${response.status}`);
  return response.json();
}

function repository(value) {
  if (
    typeof value !== "string" ||
    !/^[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+$/.test(value) ||
    value.includes("..")
  )
    throw new Error("Use an owner/repository identifier.");
  return value;
}

export async function callTool(name, args, context) {
  const run = JSON.parse(await readFile(context.runFile, "utf8"));
  if (!availableTools(run.task, context.mode).some((tool) => tool.name === name))
    throw new Error("This tool is disabled by the experiment's model or strategy permissions.");
  if (expired(context.deadline))
    throw new Error("The run budget has expired.");
  switch (name) {
    case "read_source": {
      if (typeof args.name !== "string" || !/\.(py|json|txt|md)$/.test(args.name)) throw new Error("Read a candidate source or metadata file.");
      const file = await inside(context.workspace, args.name);
      if ((await lstat(file)).size > 200_000) throw new Error("Source reads are limited to 200 KB.");
      return readFile(file, "utf8");
    }
    case "write_source": {
      if (typeof args.name !== "string" || !/^(?:[A-Za-z][A-Za-z0-9_]*\.py|candidates?\.json|plan\.json|audit\.json|model_manifest\.json|requirements\.txt|README\.md)$/.test(args.name))
        throw new Error("Write a Python module, candidate.json, plan.json, audit.json, model_manifest.json, requirements.txt, or README.md.");
      if (typeof args.content !== "string" || Buffer.byteLength(args.content) > 200_000)
        throw new Error("Source files must be smaller than 200 KB.");
      // wx refuses a pre-existing symlink; replacement requires a contained regular file.
      const destination = join(context.workspace, args.name);
      try { await writeFile(destination, args.content, { flag: "wx", mode: 0o600 }); }
      catch (error) {
        if (error.code !== "EEXIST") throw error;
        await writeFile(await inside(context.workspace, args.name), args.content);
      }
      return { written: args.name };
    }
    case "dataset_info": {
      const info = JSON.parse(await readFile(join(context.data, "manifest.json"), "utf8"));
      info.installedPackages = { ...info.installedPackages };
      for (const name of await readdir(context.packages).catch(() => [])) {
        if (!name.endsWith(".dist-info")) continue;
        const metadata = await readFile(join(context.packages, name, "METADATA"), "utf8").catch(() => "");
        const pkg = metadata.match(/^Name: (.+)$/m)?.[1], version = metadata.match(/^Version: (.+)$/m)?.[1];
        if (pkg && version) info.installedPackages[pkg] = version;
      }
      return info;
    }
    case "previous_experiments":
      return {
        selectedTrial: run.best ?? null,
        selectedScore: run.score ?? null,
        metricScores: run.metricScores ?? null,
        baseline: run.baseline ?? null,
        selectionRule: "Ablations are diagnostic. The workspace for ablation/refinement is copied from selectedTrial, even when an ablation's raw score is better.",
        trials: run.trials.map(({ id, name, phase, status, score, metricScores, detail, component, baseTrial, baseScore, impact }) => ({
          id, name, phase, status, score, metricScores, detail, component, baseTrial, baseScore, impact,
        })),
      };
    case "search_models": {
      if (typeof args.query !== "string" || args.query.length > 300)
        throw new Error("Provide a search query under 300 characters.");
      const url = new URL("https://huggingface.co/api/models");
      url.searchParams.set("search", args.query);
      url.searchParams.set("limit", "8");
      url.searchParams.set("sort", "downloads");
      url.searchParams.set("direction", "-1");
      if (args.task)
        url.searchParams.set("filter", String(args.task).slice(0, 100));
      const models = await fetchJSON(url);
      return models.map((model) => ({
        id: model.id,
        task: model.pipeline_tag,
        downloads: model.downloads,
        url: `https://huggingface.co/${model.id}`,
      }));
    }
    case "search_papers": {
      if (typeof args.query !== "string" || args.query.length > 300)
        throw new Error("Provide a search query under 300 characters.");
      const url = `https://export.arxiv.org/api/query?search_query=${encodeURIComponent("all:" + args.query)}&max_results=5&sortBy=relevance`;
      const response = await fetch(url, {
        signal: AbortSignal.timeout(20_000),
      });
      if (!response.ok) throw new Error(`arXiv returned ${response.status}`);
      return (await response.text()).slice(0, 25_000);
    }
    case "github_file": {
      const repo = repository(args.repository);
      if (
        typeof args.path !== "string" ||
        !/^[A-Za-z0-9_./-]+$/.test(args.path) ||
        args.path.includes("..") ||
        args.path.startsWith("/")
      )
        throw new Error("Provide a relative repository file path.");
      if (args.ref && !/^[A-Za-z0-9_./-]+$/.test(args.ref))
        throw new Error("Invalid revision");
      const path = `repos/${repo}/contents/${args.path}?ref=${encodeURIComponent(args.ref || "HEAD")}`;
      let file;
      try {
        const result = await runProcess("gh", ["api", path], {
          timeout: 20_000,
        });
        file = JSON.parse(result.output);
      } catch {
        file = await fetchJSON(`https://api.github.com/${path}`);
      }
      if (!file.content || file.size > 100_000)
        throw new Error("Choose a source file smaller than 100 KB.");
      return Buffer.from(file.content, "base64").toString("utf8");
    }
    case "kaggle_files": {
      if (
        typeof args.competition !== "string" ||
        !/^[a-z0-9-]{1,150}$/.test(args.competition)
      )
        throw new Error("Invalid competition identifier.");
      return (
        await runProcess(
          "kaggle",
          ["competitions", "files", "-c", args.competition],
          { timeout: 20_000 },
        )
      ).output;
    }
    case "install_package": {
      if (!packages.includes(args.name))
        throw new Error(
          "That package is not in the approved ML dependency list.",
        );
      if (typeof args.version !== "string" || !/^\d+(?:\.\d+)*(?:[a-z]+\d*)?$/.test(args.version))
        throw new Error("Choose an exact package version, such as 1.7.2.");
      await mkdir(context.packages, { recursive: true });
      const extra = ["torch", "torchvision", "torchaudio"].includes(args.name)
        ? ["--index-url", "https://download.pytorch.org/whl/cpu"]
        : [];
      // Resolve against the runtime first: --target alone reinstalls every dependency,
      // wasting the bounded /tmp filesystem even when NumPy/SciPy are already present.
      const installer = `import json, subprocess, sys
pip = [sys.executable, '-m', 'pip', 'install', '--no-cache-dir']
report = '/tmp/install-plan.json'
subprocess.run(pip + ['--dry-run', '--report', report] + sys.argv[1:], check=True)
with open(report) as f: plan = json.load(f)['install']
requirements = [entry['metadata']['name'] + '==' + entry['metadata']['version'] for entry in plan]
if requirements:
    subprocess.run(pip + ['--no-deps', '--upgrade', '--target', '/packages'] + requirements + sys.argv[2:], check=True)
else:
    print('Requested package and its dependencies are already installed.')
`;
      return (await container(context, ["python", "-c", installer, `${args.name}==${args.version}`, ...extra], {
        signal: context.signal, network: true, mounts: [[context.packages, "/packages", "rw"]], writable: context.packages,
      })).output.slice(-6000);
    }
    case "cache_model": {
      const repo = repository(args.repository);
      if (typeof args.revision !== "string" || !/^[a-f0-9]{40}$/.test(args.revision))
        throw new Error("Pin the model to a 40-character commit revision.");
      const metadata = await fetchJSON(`https://huggingface.co/api/models/${repo}/revision/${args.revision}`);
      const license = metadata.cardData?.license;
      if (!license) throw new Error("This model does not declare a license. Choose a documented model.");
      await mkdir(context.models, { recursive: true });
      await container(context, ["python", "-c",
        'import sys; from huggingface_hub import snapshot_download; snapshot_download(sys.argv[1], revision=sys.argv[2], local_dir="/models/" + sys.argv[1].replace("/", "--") + "--" + sys.argv[2], ignore_patterns=["*.py", "*.bin", "*.pkl", "*.pickle", "*.pt", "*.pth"])', repo, args.revision], {
        network: true, mounts: [[context.models, "/models", "rw"]], writable: context.models,
      });
      const record = { repository: repo, revision: args.revision, license, path: `/models/${repo.replace("/", "--")}--${args.revision}` };
      await writeFile(join(context.models, `${repo.replace("/", "--")}--${args.revision}.json`), JSON.stringify(record));
      return record;
    }
    case "list_artifacts": {
      const files = [];
      const visit = async (prefix = "", depth = 0) => {
        if (depth > 4 || files.length >= 1000) return;
        for (const file of await readdir(join(context.workspace, prefix), { withFileTypes: true })) {
          if (file.isFile()) files.push(prefix + file.name);
          else if (file.isDirectory()) await visit(prefix + file.name + "/", depth + 1);
        }
      };
      await visit();
      return files.slice(0, 1000);
    }
    default:
      throw new Error("Unknown tool");
  }
}
