# Helix ML site and runner specification

This is the main handoff document. A future coding model should read it before modifying the project.

## Hosted product revision, October 6, 2026

The user now explicitly authorized Railway hosting, a shareable portfolio project page, browser use on macOS and Windows, and removal of local Docker/setup requirements. This supersedes the historical deferred-publication scope below. Keep native Codex/Claude Code subscription sign-in, linked to persistent Helix accounts. Do not replace subscriptions with an API-key requirement.

The hosted gateway in `hosted/` uses SQLite on a persistent Railway volume for account hashes, session hashes and workspace references. Each account has a private Railway Sandbox VM running the unmodified native CLIs and existing evaluator. Provider credentials stay in that VM's native credential files and private checkpoint. No user credentials enter the base checkpoint. The gateway proxies authenticated requests with per-workspace bearer credentials. Native sign-in URLs/codes come from the CLIs, not a custom provider OAuth client. The browser first signs into its Helix account, then shows the experiment workspace before agent selection.

Idle workspaces checkpoint and shut down after five minutes without browser requests or active experiments. Live uploads/downloads and native sign-in prevent idle shutdown. Gateway restarts adopt recorded VMs. New VMs restore the user's checkpoint and current runner source; stale process locks from a snapshot are discarded only on a fresh VM. The preview has one gateway, three simultaneous VMs and a 1.2 GB training-container memory cap. Scaling, hard total storage quotas, password recovery and self-service account deletion remain future work.

Google and GitHub OAuth sign-in are optional configured methods, with explicit linking from an existing authenticated account. One-time state is bound to the initiating browser and user, PKCE protects the code exchange, and stable provider IDs identify users. Do not auto-merge accounts by name/email. Provider identity tokens are not persisted or reused for ML. Hide unconfigured methods rather than showing dead sign-in buttons.

The hosted browser needs no local CLI, Python or Docker. Local source/launcher support remains available with its existing prerequisites. Physical Mac/Windows browser acceptance is still a separate manual check; do not claim those devices were tested from Linux automation. The Railway runtime has passed real tabular/text/image/audio CPU evaluations and export checks, native auth-flow startup, and a private save/restore cycle. A one-click synthetic example configures a constrained first experiment without a filesystem upload.

## Product work, October 6, 2026

The user prioritized a workable product. LinkedIn promotion is not a product milestone. Work now focuses on useful measured experiments, reliable recovery, and independent local setup, with macOS CPU as the initial target.

Startup now discovers and writes one candidate per agent invocation, then reviews and measures it before planning the next. It no longer researches all three candidates before the first result. A fixed estimator with discovery disabled skips model research. Candidate proposals and primary source URLs are persisted incrementally; fixed user-selected estimators may have an empty source list. Existing complete research records remain usable on resume. Independent review, policy checks, full selected validation folds/seeds, export checks and the final-test rules remain enforced.

Successful readiness evidence persists for seven days, keyed to CLI version, runtime image, Node version/platform and a hash of the runner code. A forced check invalidates old evidence before it runs. Failures and expired or mismatched records cannot bypass verification. The conversation shows the actual current trial stage and completed fold fits separately from completed-trial counts.

`npm run local` checks prerequisites, prepares the runtime, installs locked dependencies, builds the UI and starts the runner. Dataset/split validation precedes agent verification on new runs. A process lock prevents simultaneous runners from opening the same data folder and recovers a dead runner's lock on restart. Saved experiments can be copied into a new editable draft with **Use these settings**.

Every scored evaluation now includes a trivial baseline fitted only on that fold's training labels, using the same metric and held-out rows. Validation/test baselines appear in experiment.json and the interface. With 0% test there is no test baseline or test score. Missing-value counts from the training pool are included in the schema shown to agents.

Stateless review calls receive the full experiment contract and explicit instructions to preserve the existing candidate except for correctness fixes. Small candidates include their current source and training schema directly in the review prompt; larger candidates use bounded file tools. Candidate plans must identify their actual model family; trial names use that implementation description rather than the research proposal's name. Repair logs retain the failure that triggered the correction.

`npm run test:acceptance` exercises real UCI bank classification and chronological daily bike regression, multiple models, six fits per candidate, refinement, recovery after a completed trial and independent inference from the downloaded ZIP. It uses native subscriptions and retains source, provenance and measured evidence under output/acceptance. See README for scope and dataset limitations. Physical macOS and independent friend onboarding remain manual acceptance work; passing Linux automation must not be described as completing those checks.

## Implementation decisions, October 4, 2026

The user authorized implementation after reviewing this handoff. The MVP targets macOS with CPU training. The final product must support macOS, Linux, and Windows, including NVIDIA GPUs and preferably Apple MPS. Hardware acceleration must be a verified capability, not an advertised placeholder. Public launch, a launch demo, Vercel deployment, and portfolio changes are deferred. The source is intended for GitHub under the MIT license. Keep the remaining product requirements, including tabular, image, text, and audio inputs.

## Interface revision — October 5, 2026

The draft composer starts at one line and grows with its text to a bounded height. Submitted and saved experiments replace the composer and draft controls with a compact agent/status bar and access to their setup. Editing a failed submission or reusing an experiment restores the draft.

The user rejected the dense form/dashboard and explanatory captions. The current interface is a dark experiment workspace with the coding-agent selector in the bottom composer and CSV/folder upload or local paths through +. A follow-up design revision adds a desktop setup rail, split visualization, protocol summary, experiment objective sheet and visible trial comparisons to distinguish Helix from a generic chatbot. On smaller screens settings open in a dialog. Model/strategy and deliverable groups expand within settings. Draft settings are editable before connection; uploads, inspection and execution still require a connected signed-in agent. Provider marks are bundled locally under public/brands. Results, actual activity and downloads appear in the conversation. Avoid reintroducing tutorial subtitles, visible configuration walls, fixed-split advice, or dashboard tabs. This supersedes the historical visual-design requirements below.

The visible split label is **Train**. The **Seed** field takes actual seed values, not their count; a comma-separated list still supports repeated evaluation. Internal development-pool identifiers and partition semantics are unchanged.

The user controls `testFraction` (0 inclusive, 1 exclusive) and `holdoutFraction` (0 exclusive, 1 exclusive). Existing runs default to .2 for compatibility. A zero test fraction produces no test rows: the final model fits all rows, export/reload is still verified, and no test score is reported. Validation remains independent during selection. Custom splits apply to independent, grouped and chronological protocols.

## Current implementation checkpoint — October 5, 2026

The local MVP is runnable. `npm ci && npm run setup && npm run build && npm start` serves the UI and authenticated API at http://127.0.0.1:4319. `npm run doctor` checks Python, Docker runtime and native subscription sign-in. The latest user explicitly requested a physically testable MVP; historical instructions below to keep execution disabled are superseded.

Agent readiness is now proved before creating or resuming an experiment. `POST /api/providers/:id/verify` makes the native agent read the schema and write Python source through Helix MCP, then uses the same isolated evaluator to train a small synthetic classifier, score predictions and verify a fresh-process Joblib reload. Sign-in alone is not verification. Checks use the native subscription but no user data or experiment budget. Failures block execution; success is cached in memory for that CLI version and runtime image, with no manual verification step in Connections. Shutdown cancels checks and removes their temporary files. GET /api/providers includes the current capability status and evidence. Both native agents have passed this check on the current Linux environment.

Run creation and pause/resume/stop are active. One run executes at a time per companion. Shutdown cancels owned work and persists checkpoints; restart resumes unfinished candidates with the remaining budget. Final evaluation cannot pause or retry after interruption. Dataset inspection, immutable random/group/time partitions, CV/seeds, target isolation, independent scoring and resource limits are implemented. The CPU runtime accepts labeled CSVs with tabular/inline-text features and explicitly referenced image, audio or text assets.

The loop researches three constrained candidates, or one when a fixed estimator is requested with discovery disabled. Permitted ablation/refinement and ensemble phases follow. Ablations record their baseline and score impact; ensemble members are copied inside the candidate workspace. Candidate dependencies are isolated and inherited from the relevant baseline. Model and strategy constraints appear in prompts, plans, reviews, tool availability and known-syntax checks. These checks do not prove arbitrary generated code compliance; arbitrary free-form constraints are still interpreted by the agent.

Python and notebook solutions are exported. Notebooks run in a fresh Jupyter kernel. Serialized models must pass their declared framework loader and reproduce predictions in a fresh process without training labels before a validation score is accepted. Native format resolves explicitly; requested formats cannot change silently. Final downloads include source, optional notebook, optional model, inference entry point, schema/configuration, partitions, dependency versions, pretrained revision/license records, experiment results and SHA-256 checksums in a ZIP. Test labels and original dataset rows remain private. The floating-point reload tolerance is `1e-6 * max(1, abs(expected))`.

Live Claude and Codex runs have passed notebook/Joblib and Python/Pickle training, respectively, including fresh-process reload, final test, pause/restart/resume and artifact download. Browser checks passed Start, pause, reload, resume, stop and bundle download at desktop and mobile widths. Runtime checks pass real CPU fits for synthetic tabular, text, image and audio inputs, CV/seeds, notebook execution, matching and deliberately mismatched exports, metric reference comparisons, final-test freezing and cancellation cleanup. Both native CLIs have passed MCP source-writing checks. See README for the full acceptance command and its subscription usage.

Remaining release work includes physical macOS acceptance, broader framework/model/format coverage, stronger enforcement for arbitrary custom code, hard aggregate disk quotas, consumer installers, hardware acceleration and cross-platform packaging. PyTorch, TensorFlow and ONNX paths have native loader/reload checks but are not claimed as framework-wide verified. Public launch, GitHub publication, Vercel and portfolio work remain deferred.

The rest of this document preserves requirements and historical findings. This checkpoint and README describe the current implementation; historical scaffold status tables are not current acceptance results.

## Scope of this handoff

This document began as a scaffold handoff and detailed specification. Implementation is now authorized under the decisions above; public deployment remains deferred. Preserve the confirmed requirements below. The working name is Helix ML.

The site should eventually deploy on Vercel and appear on the user's portfolio at `https://kavinravi.com`. Neither deployment nor portfolio modification is part of the present handoff. Do not describe the scaffold as a working autonomous ML system.

## Product intent

Build an approachable ML experiment tool inspired by Google's MLE-STAR. A user provides a dataset, prediction objective, evaluation metric, and a small number of constraints. The runner researches approaches, constructs models, evaluates them under a fixed protocol, identifies influential code components with ablation, and iterates on those components. It saves an inspectable experiment history and a reproducible final solution.

The agent must be the user's own Codex or Claude Code CLI, already authenticated with their subscription. Only those two providers belong in the initial release. The ML model being optimized is a separate concept from the coding agent. Label them distinctly in the UI.

The user liked the simplicity of T3 Code's agent selection pattern. The key correction is explicit: **the complete UI is visible before agent selection. Selecting and connecting a signed-in agent enables it.** Do not replace the workspace with an authentication splash screen, hide the configuration until selection, or ask for a Gemini/OpenAI/Anthropic API key.

The first release should support general ML tasks involving tabular data, images, text, and audio. Do not reduce the product requirement to tabular-only AutoML. General input support does not imply support for arbitrary tasks and metrics. The initial scaffold represents supervised classification and regression with a labeled CSV and optional assets. Detection, segmentation, generation, ranking, forecasting metrics, and their label schemas need explicit implementations before the site offers them.

## Requirements confirmed by the user

| Requirement             | Required behavior                                                                                                       |
| ----------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| Agent selection         | Codex or Claude Code from the user's installed, signed-in CLI. The visible UI enables after selection and connection.   |
| Small input set         | Dataset, objective, target, metric, budget, and model-selection preference. Keep detailed controls in compact sections. |
| Model search            | A checkbox labeled `Allow agent to search models`. Unchecked requires a specified model or family.                      |
| Restricted model choice | Accept a family such as `tree-based models only`, not only an exact library class.                                      |
| Broad model choice      | Checked permits task-specific discovery, with optional constraints, to maximize the selected metric.                    |
| Allowed strategies      | Explicit controls for augmentation, regularization, feature engineering, tuning, pretrained models, and ensembling.     |
| Validation              | Choose a fixed training/validation split or cross-validation. Record folds and seeds.                                   |
| Reproducibility         | Preserve splits, seeds, preprocessing, configuration, environment, and per-trial results.                               |
| Source output           | Choose a Python `.py` solution or a Jupyter `.ipynb` solution.                                                          |
| Trained model export    | Optional export with a chosen compatible format, such as joblib, pickle, PyTorch, TensorFlow, or a portable format.     |
| ML construction quality | Follow the relevant framework's official construction and evaluation practices and the machine-learning skill.          |
| Tools and integrations  | Implement the actual research, dataset, training, scoring, artifact, and connector tools the loop needs.                |
| Visual style            | Clean and simple. Avoid generic generated dashboards, decorative gradients, inflated marketing text, and fake activity. |
| Hosting                 | Prefer Vercel for the frontend; later add the project to `kavinravi.com`.                                               |

## Original scaffold state

The scaffold is a React and TypeScript frontend built with Vite. The local companion uses Node's HTTP server, filesystem, and child processes. The prototype preparation script uses Python's standard library. Docker is the proposed training boundary. There is no application database, UI component framework, authentication service, or cloud ML worker.

| Area                            | Current status                                                                                                                                               |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Workspace UI                    | Implemented with visible configuration, results, activity, artifacts, and connection views. Empty results stay empty.                                        |
| Agent dropdown and pairing      | Implemented. Checks installed CLI subscription sign-in; pairing uses a local token.                                                                          |
| Configuration controls          | Implemented, including strategy permissions, model constraints, CV/folds/seeds, source format, and model export preference.                                  |
| Task validation                 | Implemented in `local/validate.mjs`. Preserves all current configuration fields.                                                                             |
| Dataset upload                  | Implemented to the local runner. Keeps relative folder paths and refuses traversal and duplicate overwrite.                                                  |
| Experiment execution through UI | Intentionally disabled. Run creation and actions return HTTP 501. The runtime status keeps Start disabled.                                                   |
| CLI invocation adapters         | Prototype code exists for JSON event streams, native CLI authentication, and Helix MCP configuration. No live training session has verified them end to end. |
| MLE-STAR loop                   | Prototype exists in `local/engine.mjs`. Requires substantial correctness work before activation.                                                             |
| Data preparation                | Prototype uses a seeded stratified 80/20 split. It does not implement the requested CV or separate final test set.                                           |
| Metrics                         | Accuracy, binary AUROC, multiclass log loss, RMSE, and MAE scoring functions exist. Extend validation and compare with reference libraries.                  |
| Helix MCP                       | A stdio JSON-RPC server and nine tool definitions exist. Most network tools are unverified prototypes.                                                       |
| Notebook output                 | UI preference exists. Notebook generation and validation are pending.                                                                                        |
| Trained model export            | UI preference exists. Format compatibility, serialization, bundling, and reload checks are pending.                                                          |
| GPU and framework coverage      | Runtime prototype is CPU-only. PyTorch, TensorFlow, Transformers, OpenCV, and Mamba support is not verified.                                                 |
| Vercel                          | Static build configuration exists. Nothing was deployed.                                                                                                     |
| Portfolio                       | Site location identified. No portfolio changes were made.                                                                                                    |

The disabled execution guard is deliberate. Enabling it before completing the required behavior would cause the UI to advertise configuration that the prototype ignores. Treat existing engine code as material to review and simplify, not as an accepted implementation.

## Workspace and design

The scaffold uses warm off-white backgrounds, a muted green accent, fine borders, and compact typography. Source Serif is used for the page title; IBM Plex Sans for controls; IBM Plex Mono for metrics and code. Fonts currently load from Google Fonts. Bundle or self-host them if offline use is a release requirement.

The desktop layout has a narrow sidebar, a top bar with runner status and agent selection, a configuration column, and an experiment area. The sidebar lists saved experiments and links to tools/connections. Mobile stacks the configuration and results. Keep the application's visual identity when completing it unless the user requests a redesign.

Results contain a phase indicator, best validation score, final test score, completed trial count, elapsed time, and three tabs. Experiments show measured scores, durations, and outcomes. Activity shows actual agent/tool/training events. Artifacts show real downloadable files. A selected experiment explains its change and exposes its source.

Before connection, configuration remains visible and disabled. Navigation, tool descriptions, and connection controls remain usable. Opening the agent dropdown may initiate pairing if no companion is connected. A disconnected companion must disable execution and show a useful reconnection message. An installed CLI with an expired subscription sign-in must show its native terminal login command.

After connection, editing configuration is possible only when a supported selected provider reports a subscription sign-in. A real run also requires an available execution environment. In this scaffold, execution remains unavailable even when pairing succeeds.

Use visible labels, keyboard navigation, native checkboxes/selects/dialogs, meaningful empty states, and text alongside status colors. Review contrast and font sizes before release. The current compact typography and light secondary colors are a design starting point, not an accessibility certification. Tabs need full keyboard behavior. Never fill the public interface with synthetic runs, invented scores, or pretend tool activity.

## Configuration contract

`src/types.ts` defines the current task shape; `local/validate.mjs` checks it. Keep both synchronized. Do not discard settings between the UI, API, persistence, prompts, training configuration, and exported manifest.

```ts
type Task = {
  agent: "codex" | "claude";
  dataset: string;
  target: string;
  objective: string;
  metric: "accuracy" | "auroc" | "log_loss" | "rmse" | "mae";
  minutes: number;
  trials: number;
  searchModels: boolean;
  model: string;
  output: "py" | "ipynb";
  exportModel: boolean;
  exportFormat:
    | "native"
    | "joblib"
    | "pickle"
    | "pytorch"
    | "torchscript"
    | "keras"
    | "savedmodel"
    | "onnx";
  validation: "holdout" | "cv";
  folds: 3 | 5 | 10;
  seeds: number[];
  policy: {
    augmentation: boolean;
    regularization: boolean;
    features: boolean;
    tuning: boolean;
    pretrained: boolean;
    ensemble: boolean;
  };
};
```

Default assumptions in the scaffold are accuracy, a 30-minute budget, 12 candidate trials, model search enabled, three-fold CV, seed 42, Python output, and native model export. Augmentation defaults off. Other strategy permissions default on. These are product defaults chosen for the scaffold, not user-mandated values.

The current server validator accepts 1 to 1440 minutes, 3 to 100 trials, and one to five unique unsigned 32-bit seeds. A trial means a candidate code/configuration evaluated under the complete selected protocol, not a single fold fit. Repairs, fold fits, research, caching, final evaluation, and export still consume wall-clock time. Show the total training workload before starting when CV multiplies it substantially.

### Model search and constraints

When model search is unchecked, `model` is required. Exact requests such as `XGBoost classifier` and family requests such as `tree-based models only` are valid. The loop may compare permitted implementations within that family and optimize them subject to the strategy controls. It must not introduce neural networks, linear models, or a forbidden ensemble just because their scores might improve.

When model search is checked, `model` becomes an optional restriction. An empty restriction allows task-specific model discovery. A checked search box and a restriction such as `CPU only; no pretrained weights` still constrain candidates.

Unchecked search does not forbid documentation lookup for the specified model. Disable broad discovery tools and candidate changes outside the resolved constraint. Preserve both the user's text and the resolved machine-readable constraint. Reject ambiguous or contradictory constraints with a specific question before starting dependent work. Do not silently broaden the model family.

### Allowed strategies

Permissions grant ability. They do not require the agent to use a technique.

| Control             | Checked                                                                                                       | Unchecked                                                                                                                    |
| ------------------- | ------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| Augmentation        | Permit valid input transformations, resampling, or synthetic training examples. Log methods and random seeds. | Preserve original training examples. Do not add synthetic samples or randomized augmentation.                                |
| Regularization      | Permit explicit penalties, dropout, weight decay, and other declared regularization.                          | Do not introduce or tune regularizers. Resolve required nonzero framework defaults before accepting a model.                 |
| Feature engineering | Permit derived features, learned feature selection, and task-specific representations.                        | Allow required decoding, imputation, encoding, and normalization. Do not introduce predictive derived features or selection. |
| Parameter tuning    | Permit a bounded hyperparameter search within the candidate budget.                                           | Use recorded fixed settings. Do not disguise a search as repeated otherwise-identical candidates.                            |
| Pretrained models   | Permit approved pretrained weights, embeddings, and fine-tuning. Record model revision and license.           | Train from scratch. Do not use cached weights or precomputed pretrained embeddings.                                          |
| Ensembling          | Permit permitted-model averaging, voting, blending, or stacking.                                              | Skip initial merging and final ensemble phases. Export a single permitted solution.                                          |

An unchecked regularization box needs careful semantics. Some estimators and optimizers include regularization by default. The intended requirement is a model with no added or adjustable regularization. If a requested model cannot meet that requirement, explain the conflict instead of silently retaining a default penalty. Apply the same principle to feature extractors that hide pretrained weights.

Represent permissions in the resolved plan, tool availability, code review, and recorded candidate configuration. A prompt alone is insufficient enforcement. Static checks also cannot prove that arbitrary generated Python obeys every policy. State the supported checks precisely, reject known violations, and retain a reviewable audit.

## Dataset and evaluation protocol

The scaffold accepts a local CSV or folder. Folder discovery currently looks for `train.csv`, `labels.csv`, then `metadata.csv`. A folder can contain text, image, or audio assets referenced by relative paths in the CSV. Preserve row IDs and asset paths. Nested symlinks are excluded by the preparation prototype. Do not copy unrelated neighboring directories just because they share a CSV's parent.

Before model research, inspect schema, target type, missing labels, repeated rows, class imbalance, duplicated entities, asset availability, and usable sample counts. Produce a dataset manifest and a stable content fingerprint. Tell the user which task and metric combinations are supported. Do not interpret arbitrary media folders as labeled supervised datasets without labels.

Create a final test partition once. Persist its row IDs and fingerprint. The initial proposed fraction is 20 percent, leaving 80 percent for model development. This fraction is a design assumption that may become an advanced control. Final test data must never enter research, model selection, early stopping, ablation, tuning, or fold statistics.

For fixed-split evaluation, split the development partition into training and validation. The scaffold's label proposes 20 percent of development data for validation. For CV, split only the development partition into the requested folds. Classification normally needs stratification; regression normally uses shuffled K-fold. Grouped, repeated-entity, or time-ordered data needs a group-aware or chronological strategy. Add explicit group/time inputs or block an unsafe default. Do not claim random CV is universally correct.

Materialize the exact same validation rows/folds for all candidates. For multiple seeds, distinguish split seeds from model/loader seeds. A simple initial rule is a single final test split using the first seed, with repeatable development folds for each listed seed and corresponding model seeds. Record that rule. Changing the evaluation protocol starts a new run.

Every fold must fit imputation, encoding, scaling, feature selection, and learned transformations using that fold's training rows. Augmentation applies only to training data. Validation and test transforms must be deterministic unless the user explicitly permits and the manifest records a defined inference-time strategy. Do not train on validation labels through helper files or mounted sibling folders.

The trusted evaluator computes metrics from ordered predictions and target values unavailable to candidate training. The prototype's label files are outside its training mount, but the native agent runs as the user's host account and can potentially read them. **Filesystem location alone is not test-label isolation.** Design and verify a real read boundary before claiming the agent cannot access test targets. Restrict candidate/agent tools and use an evaluator process or OS identity with protected storage as appropriate. Do not give the agent the original unsplit dataset path after preparation if it defeats that boundary.

Persist fold IDs, seed IDs, sample counts, scores, mean, standard deviation, and failures. Confidence intervals need a stated method and assumptions. Correlated CV folds are not independent observations. The UI currently has a placeholder for an approximate interval with this caveat; do not report a naive fold interval as a guarantee of statistical validity. Cross-validation usually improves the assessment over one split, but does not guarantee identical results across seeds.

Compare candidates using the same aggregation and metric direction. Accuracy and AUROC maximize; log loss, RMSE, and MAE minimize. A failed fold must not disappear from the average. Reject incomplete candidates or apply a documented rule. Small changes within run variation should remain visible as uncertainty.

After selection, freeze the winning approach and evaluate the final test set once. Persist the result transactionally so resume cannot trigger repeated test-driven tuning. The user may optionally request a separate deployment refit on all labeled rows afterward. Keep its training data and identity distinct from the model that produced the test score. Never claim that the all-data refit itself has the earlier independent test score.

## MLE-STAR-inspired loop

Google's system combines retrieved model implementations, measured baselines, model merging, ablation-guided refinement, and ensemble search. The main references are the [Google Research description](https://research.google/blog/mle-star-a-state-of-the-art-machine-learning-engineering-agents/) and [MLE-STAR paper](https://arxiv.org/html/2506.15692v3).

Use the paper's algorithmic ideas. Helix's coding agents, permission controls, local runtime, and output packaging are product adaptations. Do not promise Google's benchmark performance or describe the app as an exact reproduction.

The intended sequence is:

1. Validate the task, permissions, dataset, runtime, and budget. Persist a complete run specification and evaluation protocol.
2. Research a small set of approaches, with real source URLs and an inexpensive baseline. Constrain the set when model search is disabled.
3. Generate and review each candidate, then let the runner train and score it. The coding agent cannot invent its own measured result.
4. If ensembling is allowed, try a simple initial blend of complementary measured approaches.
5. Ablate one component of the current best solution while preserving everything else. Track the change in validation performance relative to that same solution.
6. Select an influential component and run a bounded inner refinement loop focused on its code block. Use actual prior results as feedback.
7. Keep improvements under the fixed protocol. Repeat ablation/refinement within budget and permitted strategies.
8. If allowed and feasible, search an ensemble strategy using validation or out-of-fold predictions. Stacking's meta-model must train on out-of-fold predictions, never on predictions from base models trained on those same rows.
9. Freeze selection, perform the one final test evaluation, export the source and optional model, verify reloading, and save the reproducibility report.

Debugging can repair a failed candidate without changing the task or evaluation. Check leakage, dependencies, class order, tensor shapes, asset paths, actual data usage, and policy compliance before accepting it. Ablation is a diagnostic phase, not automatically a winning replacement. A resource-limited data subsample must be stable across candidates and recorded, and its score must not masquerade as full-data evaluation.

The present prototype researches three candidates, attempts an initial merge, alternates ablation and refinement, and ends with an ensemble. Its schedule ignores strategy permissions and its preparation ignores CV/seeds. Replace those behaviors before connecting run creation. Reserve budget for final evaluation and export instead of spending all time on research/refinement and marking an incomplete deliverable as complete.

## Source output and framework construction

The requested `.py` or `.ipynb` choice is a source deliverable, separate from trained model serialization. Keep training and inference code structured, explicit, and rerunnable. Avoid a long script assembled from unrelated patches or a notebook that works only because cells ran out of order.

A Python solution should have configuration, data preparation, model construction, training, evaluation, and inference functions. Use a `main` entry point and CLI arguments. A notebook should contain task/data context, configuration, preparation, model construction, training, evaluation, interpretation, and export sections. Run it from a fresh kernel in cell order. Do not include hidden state, secrets, absolute developer paths, or invented output cells. Reuse a clean notebook template and the jupyter-notebook skill if available.

The current prototype contract uses `train.py` with `--train`, `--validation`, `--metadata`, `--models`, and `--output`. It predicts in input order. Accuracy uses original class labels, binary AUROC uses probabilities for `classes[1]`, multiclass log loss uses probability arrays in the recorded class order, and regression uses finite numbers. Extend this contract with an explicit seed and machine-readable resolved configuration. Do not let the agent change the trusted scorer.

| Framework        | Construction and evaluation requirements                                                                                                                                                                                                                                  |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| scikit-learn     | Use `Pipeline` and `ColumnTransformer` as appropriate. Fit transforms inside each training fold. Record estimator parameters, class order, and random state. Export the entire preprocessing/model pipeline.                                                              |
| PyTorch          | Use explicit modules and training loops or an appropriate trainer. Set train/eval modes correctly, reset gradients, match loss and target shapes, handle device placement, and disable gradients for evaluation. Save architecture/config with weights and preprocessing. |
| TensorFlow/Keras | Use supported model construction and `fit`/evaluation APIs. Set seeds and deterministic options where practical. Define losses/metrics consistently. Keep `.keras` training serialization distinct from SavedModel inference export.                                      |
| OpenCV           | Declare image decoding, channel order, dtype, resize, and normalization. Handle failed reads. Match preprocessing during reload and inference. Treat OpenCV as an input/vision library unless a specific OpenCV model is requested.                                       |
| Transformers     | Pin pretrained model/tokenizer/processor revisions. Use proper collators, masks, padding, label mappings, and the task's training head. Record Trainer configuration and save processors alongside weights. Respect the pretrained permission.                            |
| Mamba            | Follow the selected official implementation's installation and tensor shape requirements. Record architecture and dependencies. Detect actual device/kernel support and do not promise GPU kernels on a CPU-only runtime.                                                 |

The machine-learning skill was explicitly requested. The local copy read for this handoff is `/home/kavin-ravi/.agents/skills/machine-learning/SKILL.md`. A future environment may not have that path. Carry these requirements in the implementation regardless: strict train/validation/test separation, appropriate stratification, reproducible seeds, confidence/variation reporting, logged parameters/metrics/artifacts, and SHAP or permutation-based interpretation for models beyond a baseline. For images/text/audio, choose a suitable attribution or perturbation method and record its limitations. Interpretation must not drive tuning on the final test set.

Seed Python, NumPy, framework RNGs, data loaders, samplers, and augmentation. Record nondeterministic operations and device/library differences. Deterministic settings can reduce throughput and cannot guarantee bitwise equality across hardware or library versions. Do not present the seed list as a complete reproducibility guarantee.

Use current primary documentation during implementation:

- [scikit-learn common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html)
- [PyTorch reproducibility](https://docs.pytorch.org/stable/notes/randomness.html)
- [PyTorch saving and loading](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)
- [Keras training methods](https://keras.io/guides/training_with_built_in_methods/)
- [Keras serialization and saving](https://keras.io/guides/serialization_and_saving/)
- [OpenCV Python tutorials](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)
- [Transformers Trainer](https://huggingface.co/docs/transformers/en/main_classes/trainer)
- [Transformers data collators](https://huggingface.co/docs/transformers/en/main_classes/data_collator)
- [Official Mamba repository](https://github.com/state-spaces/mamba)

## Trained model export

Provide an `Export trained model` checkbox and a format dropdown. Source output remains available when export is unchecked. The prototype currently tells candidates to save fitted models unconditionally; change that contract and artifact filtering before honoring this option in execution.

The scaffold dropdown lists formats to capture the requirement. The finished UI must filter or explain incompatible choices after model selection. An explicit format is a constraint on candidate selection; it cannot silently change to another format at the end. If conversion fails, preserve the validated solution and report the export failure without calling the entire deliverable complete.

| Format                         | Intended use                                                               | Required companion artifacts/checks                                                                                                     |
| ------------------------------ | -------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| Native framework format        | Default. Select a documented native format for the actual model/framework. | Record the resolved format and exact reload command.                                                                                    |
| Joblib `.joblib`               | Python estimators and compatible preprocessing pipelines.                  | Dependency versions, full pipeline, input schema, prediction reload check.                                                              |
| Pickle `.pkl`                  | Explicitly requested Python serialization.                                 | Versions and full object dependencies. Loading untrusted pickle/joblib can execute code; show that warning at the import/load decision. |
| PyTorch state dictionary `.pt` | PyTorch weights, including compatible Transformers or Mamba models.        | Architecture source/config, processor/tokenizer, label map, safe reload instructions, prediction check.                                 |
| TorchScript `.pt`              | Explicitly requested compatible PyTorch inference serialization.           | Verify support in the selected PyTorch version and actual model. Do not assume every graph can trace/script.                            |
| Keras `.keras`                 | Native Keras model serialization where supported.                          | Preprocessing, registered custom objects, dependency versions, reload check.                                                            |
| TensorFlow SavedModel          | TensorFlow serving/inference export.                                       | Explicit signatures, preprocessing, directory bundle, inference check.                                                                  |
| ONNX `.onnx`                   | Supported portable inference graphs.                                       | Opset and runtime version, preprocessing, conversion validation, numerical/prediction equivalence tolerance.                            |

Do not pickle a complete arbitrary PyTorch object as the default PyTorch export. Do not export only neural weights while omitting the architecture needed to rebuild the model. Do not equate a portable model graph with preprocessing or tokenizer portability.

Bundle a manifest, prediction entry point, input schema, class/label order, preprocessing assets, locked dependencies, source, and checksums with the model. Ensembles need all members, weights/meta-model, and shared preprocessing. Directory formats need a downloadable archive. Generate archives from an explicit artifact list and reject symlinks/traversal.

Before presenting export as successful, load it in a fresh runtime and compare predictions on a stored nonsensitive development fixture or a documented local fixture. Record tolerance for floating-point conversion. Include an example that loads the artifact and predicts without retraining or accessing test labels.

## Architecture and subscription integration

Use a Vercel-hosted static UI plus a local companion. The browser sends task configuration to a paired loopback service. That companion runs the user's native CLI and local training runtime. Dataset bytes, agent authentication, generated model files, and experiment state remain local by default.

```text
Browser UI on Vercel or localhost
  -> paired loopback HTTP companion
      -> native Codex or Claude Code CLI
          -> native research tools + Helix stdio MCP
      -> isolated Python training runtime
      -> trusted evaluator and local artifact store
```

The website alone cannot use a visitor's local CLI sign-in. A local companion is necessary for this design. Package it so installation is much simpler than the original MLE-STAR setup. The current `npm ci` plus `npm run bridge` workflow is a developer scaffold, not the final consumer installation experience.

Keep the native CLI's subscription authentication. Do not build a custom Claude.ai/ChatGPT login, copy OAuth credentials into Vercel, expose them to the browser, or exchange subscription tokens through a Helix backend. Strip API-key/provider override environment variables when the intent is subscription-backed execution. Detect and reject API-key-only sign-in in this product mode.

The adapter prototype uses `codex exec --json` with workspace-write sandboxing and a configured stdio MCP server. It uses `claude -p --output-format stream-json --verbose` with an explicit tool list, a noninteractive permission mode, and Helix MCP configuration. Claude's `--bare` mode is unsuitable for this subscription design. Verify all flags against the installed CLI versions before implementing the production adapters.

For Claude Code, the documented integration route is the native unmodified binary with each user's own sign-in. Review the current [headless documentation](https://code.claude.com/docs/en/headless), [authentication documentation](https://code.claude.com/docs/en/authentication), and [legal/compliance documentation](https://code.claude.com/docs/en/legal-and-compliance) before release. For Codex, review the [noninteractive documentation](https://learn.chatgpt.com/docs/non-interactive-mode) and [authentication documentation](https://learn.chatgpt.com/docs/auth). These references do not imply unlimited subscription usage. Preserve native quota/rate-limit errors and offer pause/resume.

Existing user MCP configuration is an integration goal, subject to the run's tool permissions. Do not blindly grant every global MCP server access to datasets or permit a configured server to bypass constraints. Define supported inheritance and session-scoped configuration, and inspect tool permissions before invoking them. The prototype loads user configuration without this review.

The local server binds to `127.0.0.1`. It checks Host and Origin, requires a random pairing token on API requests, and stores the token with owner-only permissions. Browser pairing uses session storage and authenticated requests; downloads use a bearer header rather than query-string tokens. Pairing grants access to the companion's local operations, so retain explicit origin binding.

Verify deployed HTTPS-to-loopback access in current Chrome, Safari, and Firefox. Browser local-network permissions, CORS/preflight, and platform restrictions can differ. Prefer local same-origin UI as the fallback. Do not promise that hosted pairing works merely because local development does. Ensure Vercel response policy permits only the intended loopback connections.

## Tools, connectors, and MCP

Avoid an ornamental tools marketplace. Each integration should have an actual callable implementation, a clear permission boundary, and honest status. The initial relevant integrations are native web search, dataset inspection/import, model/research discovery, source lookup, dependency/model caching, isolated training, trusted scoring, and artifact export.

The current nine Helix MCP tools are:

| Tool                   | Prototype operation                                    | Remaining requirements                                                                                                             |
| ---------------------- | ------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------- |
| `dataset_info`         | Read the prepared manifest.                            | Add schema, assets, quality checks, split protocol, and useful samples without labels that should be hidden.                       |
| `previous_experiments` | Return recorded trial scores and changes.              | Include resolved configs, fold/seed variation, costs, and component deltas.                                                        |
| `search_models`        | Search public Hugging Face models.                     | Enforce model-search and family permissions; record revisions/licenses and verify network behavior.                                |
| `search_papers`        | Search arXiv's API.                                    | Parse primary-source results, handle rate limits, preserve attribution.                                                            |
| `github_file`          | Fetch a small source file through `gh` or public API.  | Pin revisions, handle authentication/rate limits, track licenses; never publish private source by accident.                        |
| `kaggle_files`         | List competition files using local Kaggle CLI.         | Check actual sign-in and access. This does not download a dataset yet.                                                             |
| `install_package`      | Install allowlisted dependencies into a run directory. | Pin versions, resolve compatibility, log hashes, bound jobs, kill containers on interruption.                                      |
| `cache_model`          | Download a public Hugging Face repository snapshot.    | Enforce pretrained permission, pin revisions, support native local auth for gated repos when allowed, avoid remote-code execution. |
| `list_artifacts`       | List files in the current candidate directory.         | Add approved export/bundle metadata, checksums, and secure filtering.                                                              |

Add actual dataset acquisition through Hugging Face/Kaggle when needed for the supported input flow. Local upload/path input is the scaffold's working data entry. Competition data downloads require access and accepted competition terms through the user's own account. Do not claim those connectors are fully implemented because file-list or model-search tools exist.

Use a small tested MCP implementation or a maintained official SDK if it reduces compatibility work. Verify initialization/version negotiation, tool schemas, error results, stdout discipline, concurrent calls, and cancellation with both real clients. The prototype's protocol version is fixed and its requests run serially.

## Runtime, persistence, and API

Training runs must have bounded CPU, memory, disk, process count, and time. The proposed Docker defaults are two CPU cores, 3 GB memory, no network, a read-only root filesystem, dropped capabilities, and a bounded temporary directory. Keep development/test labels outside the generated code's mounts. Mount model and dependency caches read-only during training.

Model/dependency downloads need network access in separate approved jobs. Label every run-owned container and enforce an internal deadline, so killing the Docker client cannot leave an orphaned job. On pause, stop, deadline, server shutdown, or restart, stop all owned jobs and preserve completed results. Never clean up unrelated containers.

The prototype keeps state under `.helix/runs/<run-id>/` with data, evaluator files, research, trials, final outputs, model cache, and package cache. Run metadata uses atomic temporary-file replacement. One active run per local companion is an adequate initial limit; add a queue only when needed. Store the executing promise and await its cleanup/checkpoint during shutdown. Resuming restarts only unfinished work and charges actual active wall-clock time.

Expected endpoints are:

| Endpoint                              | Scaffold behavior / target behavior                                        |
| ------------------------------------- | -------------------------------------------------------------------------- |
| `GET /health`                         | Public local name/version only.                                            |
| `GET /api/providers`                  | Authenticated installed/subscription sign-in checks.                       |
| `GET /api/tools`                      | Authenticated integration status; runtime reports execution pending today. |
| `GET /api/runs`                       | Authenticated persisted history.                                           |
| `POST /api/runs`                      | HTTP 501 today; later validate, freeze configuration, and start a run.     |
| `GET /api/runs/:id`                   | Authenticated snapshot.                                                    |
| `POST /api/runs/:id/action`           | HTTP 501 for existing runs today; later pause, resume, or stop.            |
| `GET /api/runs/:id/artifacts`         | Authenticated artifact listing.                                            |
| `GET /api/runs/:id/file?path=...`     | Authenticated contained-path artifact download.                            |
| `POST /api/datasets`                  | Create an authenticated local upload directory.                            |
| `PUT /api/datasets/:id/file?name=...` | Stream a contained relative file. Current per-file ceiling is 2 GB.        |

Future state must distinguish queued, research, candidate generation/review/training/evaluation, paused, stopped, failed, finalizing, and completed. A failed refit/export/test stage needs a specific outcome. The prototype can mark a run complete after an unsuccessful final refit; replace that behavior with accurate completion semantics.

## Original engine gaps to resolve before activation

Read every caller before repairing these. A frontend checkbox without backend behavior is unfinished.

- `prepare.py` currently makes one 80/20 split with seed 42. It does not consume the selected validation/folds/seeds or reserve an independent test set.
- `engine.mjs` does not honor model-search restrictions, any strategy checkbox, source output choice, or model export preference.
- Research always requires three candidates. The restricted-model case needs three permitted approaches or a smaller justified schedule, not three forbidden families.
- The prototype's final refit uses every labeled row and predicts the old validation rows. It does not produce an independent final test score.
- Candidate scripts can access prior scripts on the host, but the training container does not mount prior solutions for an ensemble. Resolve referenced artifacts explicitly.
- Only `train.py` is copied when seeding/refitting a candidate. Helper modules, model assets, preprocessing state, and architecture files can be lost.
- The agent's `audit.json` is self-reported. Independent protocol checks and known-policy checks are still required.
- Dependency/model caching containers lack complete ownership/cancellation cleanup. Versions and model revisions are not locked.
- Claude's allowed tools and user config inheritance need filesystem restrictions and permission verification. Codex's local read scope also needs review.
- Agent process success must reflect terminal structured events, not merely a zero process exit. Handle provider errors, stream truncation, quota limits, and malformed output.
- Preparation copies neighboring non-code assets. Restrict copying to declared dataset assets with quotas and an explicit manifest.
- Logs redact only a small set of token patterns. Avoid including credential-bearing child-process output at all, and verify logs/artifacts for account or secret disclosure.
- Run persistence, interruption, restart recovery, and final test idempotency need a real training integration check.
- `.py`/`.ipynb` export, model format selection, reload verification, and complete artifact packaging are absent.
- Installed Kaggle CLI is not authenticated access. Public source connectivity is not a completed connector integration.
- The CPU runtime cannot satisfy arbitrary deep model requests within a short budget. Detect feasibility and explain required hardware instead of quietly substituting a model.

## Repository map and development commands

| Path                                  | Purpose                                                                              |
| ------------------------------------- | ------------------------------------------------------------------------------------ |
| `src/App.tsx`                         | Visible workspace, configuration, pairing, results, downloads, and connection views. |
| `src/styles.css`                      | Current visual design and responsive layout.                                         |
| `src/types.ts`                        | Frontend task/run/trial/provider/tool contract.                                      |
| `src/api.ts`                          | Authenticated loopback requests and pairing URL/token validation.                    |
| `local/server.mjs`                    | Local HTTP API, origin/Host/auth checks, uploads, history, disabled execution guard. |
| `local/validate.mjs`                  | Server task validation and contained file paths.                                     |
| `local/agents.mjs`                    | Subscription status detection and native CLI adapter prototypes.                     |
| `local/engine.mjs`                    | Unfinished experiment loop and persistence prototype.                                |
| `local/prepare.py`                    | Seeded fixed-split CSV/asset preparation prototype.                                  |
| `local/metrics.mjs`                   | Trusted metric scoring prototype.                                                    |
| `local/mcp.mjs`, `local/tools.mjs`    | Stdio protocol and research/cache connector prototypes.                              |
| `local/process.mjs`                   | Bounded child processes and process-group interruption.                              |
| `local/Dockerfile`, `local/setup.mjs` | Optional CPU training image and build helper.                                        |
| `local/check.test.mjs`                | One runnable scaffold integration check.                                             |
| `vercel.json`                         | Static frontend build and response headers.                                          |
| `README.md`                           | Short local preview instructions.                                                    |

```sh
npm ci
npm run dev
npm run bridge -- --origin http://127.0.0.1:5173
npm run build
npm test
```

Node 22.13 or newer is required. Python 3 is needed for the preparation check. Docker and either signed-in CLI are needed later for actual experiment execution; neither is needed to view the UI. `npm run bridge` starts the pairing/status/upload companion now. `npm run setup` builds the runtime prototype, but does not remove the execution guard.

The workspace's `.git` is an environment-owned placeholder rather than a normal initialized checkout. A future implementation should inspect repository state and choose an authorized repository location before publishing. Do not overwrite the placeholder.

## Suggested implementation order

### Checks completed for this handoff

`npm run build` passed TypeScript checking and the production Vite build. `npm test` passed the single scaffold integration check. The Docker runtime image built successfully, but no agent-generated model was trained or scored.

Browser checks at desktop and 390-pixel mobile widths verified the initial visible/disabled configuration, pairing with the actual local companion, signed-in Codex and Claude Code selection, the required model-family input when search is off, export-format disabling when export is off, and no horizontal page overflow on mobile. The execution button remained disabled. Screenshots are under `output/playwright/` and are ignored by version control.

These checks cover the scaffold behavior. They do not validate the autonomous experiment loop, network connector tools, model exports, full accessibility, or a deployed HTTPS pairing workflow.

### Work for the next model

1. Read this document and review the current runnable MVP. Establish a short regression check for configuration persistence and pairing behavior.
2. Build dataset manifests, immutable development/test partitions, fixed-split/CV/multi-seed evaluation, and a trustworthy target-access boundary. Check them without calling an agent.
3. Verify native CLI invocation and MCP tools with both installed agents using small authorized tasks. Resolve local-only sign-in, filesystem restrictions, error streams, and policy-aware tool availability.
4. Implement a single constrained candidate end to end. Independently score it, preserve all settings, record versions/seeds, and verify pause/resume/deadline cleanup.
5. Extend that path with research, measured baselines, component ablation/refinement, and optional ensembles. Keep budget and permission behavior consistent through every phase.
6. Add notebook source generation, model exports, complete bundles, interpretation, and fresh-runtime reload checks. Verify failure reporting and one final test evaluation.
7. Exercise a small tabular, image, text, and audio classification/regression case using actual assets. Verify supported frameworks on declared hardware. Show unsupported combinations clearly.
8. Finish UI accessibility, keyboard behavior, contrast, browser pairing, consumer setup, and error recovery. Remove prototype status wording only when the corresponding functionality works.
9. Deploy the static UI on Vercel and test its real HTTPS pairing workflow. Add a truthful project entry to the portfolio after the deployment URL is known.

## Acceptance checks for the future implementation

- An unsigned visitor sees the complete UI. Selecting a dropdown option alone never grants access to a missing or signed-out CLI.
- Pairing does not read, transmit, or store the user's provider credentials. An unpaired or unapproved origin cannot start a run, upload data, or download artifacts.
- Both supported native CLIs complete a tiny real task through Helix, using their native subscription sign-in. No API-key fallback is silently billed.
- A model-family restriction stays intact through baselines, refinement, and export. General model search can choose a permitted approach using actual research sources.
- Each unchecked strategy changes the allowed plan/tools/candidate audit. No initial merge or final ensemble runs when ensembling is off.
- Every candidate uses the same recorded evaluation protocol. Fold preprocessing uses training rows only. Test rows never enter tuning or interpretation-driven selection.
- Multiple seeds produce recorded per-seed results and aggregate variation. Failed fits cannot improve an average by disappearing.
- The final test result is independently computed once after selection and survives pause/restart without another selection round.
- Pause, stop, timeout, quota error, and shutdown leave no owned processes/containers running and preserve completed trials.
- Python output runs in a fresh environment. Notebook output executes from a fresh kernel in order. Both include a complete configuration and reproducibility record.
- Each offered model format is compatible, bundles required preprocessing/architecture, reloads, and reproduces inference within a recorded tolerance. Export disabled produces source/report without promising a serialized model.
- Public UI has no invented scores or source references. Actual failures identify the failed phase and retain the best usable artifacts.
- A Vercel build succeeds without starting local processes in cloud functions. Hosted pairing works in the documented browsers or offers a working local fallback.
- The portfolio points to a real deployed URL and accurately distinguishes implemented behavior from roadmap work.

## Later Vercel and portfolio integration

Deploy only the static frontend build, currently `npm run build` producing `dist`. Do not move a user's CLI credentials or multi-hour local training into Vercel functions. Recheck current Vercel/browser behavior when implementing hosting rather than relying on old function-duration or WebSocket assumptions.

The local portfolio repository was identified at `/home/kavin-ravi/CodingStuff/kavinravi-site`. Its README identifies `lib/content.ts` as the source of project/research descriptions. It uses Next.js and already has project listing/detail routes. Inspect its current content shape, repository instructions, and deployment linkage before making a portfolio change. This path may differ in a later environment.

The eventual project entry should explain the actual problem and behavior, identify its MLE-STAR inspiration, link to the working app and source when available, and use a real screenshot. Keep the main product interface an experiment workspace. The user did not request a separate marketing site.

## Prompt to give the next coding model

> Read HELIX_ML_SPEC.md and README.md, then inspect the scaffold. Implement Helix ML according to the confirmed requirements. Preserve the complete visible UI and enable it through the user's signed-in Codex or Claude Code CLI. Prioritize trustworthy evaluation, enforceable model/strategy constraints, reproducibility, and reloadable source/model artifacts. The prototype engine is incomplete and UI execution is deliberately disabled. Preserve the working MVP and extend coverage before making broader release claims. Continue with Vercel deployment and portfolio integration only in the implementation phase requested by the user. Report real checks and remaining limitations accurately.


## October 2026 update: prompt review and unsupervised tasks

The chat message now specifies the learning task, metric, and model preference. The selected native agent returns a proposal based on the message, CSV column names, and settings; Helix validates it and displays it before the user confirms training. Ambiguous requests ask for a detail instead of launching. Model and metric selectors are removed from draft settings. Budgets, strategy permissions, split controls, target/exclusion overrides, and exports stay in Settings.

Numeric CSV clustering uses held-out cluster assignments, scored with silhouette or Davies–Bouldin in a fixed training-fitted median-imputation/standardization reference space. Models must assign every unseen row; noise labels and over 1,000 clusters are unsupported. Dimensionality reduction uses held-out transformed coordinates and trustworthiness. Users choose either a fixed dimension count for PCA/UMAP/other transformers, or a PCA cumulative explained-variance target. Variance mode fixes preprocessing and full-solver, non-whitened PCA and independently verifies selected dimensionality and output geometry. All learned steps fit training rows only. Metrics use the same seeded sample of at most 1,000 held-out rows per fold. Unsupervised ensembles and raw nonnumeric inputs are not supported in this iteration.

Final results include a bounded scatterplot and representation.csv with original zero-based CSV row indices and cluster IDs or coordinates. With no test split, output covers all rows; otherwise it covers the test partition. Follow-ups output the training pool and do not reread the test partition. Full-data refits have no in-sample score. The ordinary final-test freeze, source checks, package isolation, and fresh-process export/reload checks remain in force.

## October 2026 update: dataset scaling

The 100,000-row and 128 MiB CSV preparation caps are removed. Inspection, hashing, snapshotting, and fold materialization stream the CSV; duplicate checks use temporary SQLite storage. Compact class/group codes and immutable protocol-v2 split indices remain in memory under an estimated preparation budget. Previous partition hashes remain compatible. Storage, schema, class-count, and asset limits still apply.

Candidate admission estimates data buffers, output width, known model-specific allocations, and the trusted unsupervised scorer against the same RAM budget used by the training container. Chunked incremental candidates can use smaller data buffers. Static estimates cannot predict arbitrary generated code; Docker enforces the final limit, reports out-of-memory failures, and preserves completed trials. Defaults are 512 MiB preparation and 1,200 MiB hosted / 3,072 MiB local training, adjustable with HELIX_PREP_MEMORY_MB and HELIX_TRAIN_MEMORY_MB on appropriately provisioned workers. This does not make every estimator or scorer out-of-core, and training is never silently subsampled. The opt-in scaling check exercises one million rows, a CSV above the former size cap, and model/feature/fold memory limits.
