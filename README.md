# Helix ML

Run ML experiments with your own Codex or Claude Code subscription. Upload labeled data, compare independently measured candidates, and download the selected source and trained model.

## Open in your browser

[Open Helix](https://helix-web-production-03c9.up.railway.app) · [Project page](https://kavinravi.com/projects/helix-ml)

Create a Helix account with an enabled Google/GitHub sign-in option or a username and password, choose Codex or Claude Code, and follow the native agent's sign-in link. Codex uses a device code; Claude Code returns a confirmation code to paste into Helix. Your account restores the same private workspace on later visits. Choose **Try an example** to load a small CSV and working experiment settings, or attach your own data.

The hosted app runs on Railway. Mac, Windows and Linux users need a browser and their agent subscription, with no local terminal, Python or Docker installation. Each account gets an isolated Linux VM with its own native agent credentials and experiment files. The shared base image contains no user credentials. Helix account passwords are hashed with scrypt; sessions use HttpOnly cookies. Provider credentials remain in the native CLIs' files in that user's private workspace and saved checkpoint. Helix does not collect provider passwords or API keys.

This CPU preview admits three active workspaces. Each fit has two CPU cores and a 1.2 GB memory ceiling, leaving memory for the agent and server. Closing the browser lets an active experiment finish. After five idle minutes, an inactive workspace is checkpointed and shut down; signing in restores it. Paused runs remain resumable. A machine failure before a checkpoint can lose changes since the last saved checkpoint. Dataset metadata, objectives, generated code and tool results reach the selected agent provider. Uploaded files and native login state are stored on Railway.

Current verification covers real Railway CPU fits, model reload, private workspace save/restore, account separation and browser flows. Physical Mac and Windows acceptance still requires testing on those machines. NVIDIA/MPS acceleration and desktop installers remain future work.

## Self-host the web app

Deploy the root `Dockerfile` as a Railway service with a persistent volume at `/data`, one replica, and `/health` as its healthcheck. Install the separate hosted dependencies with `npm ci --prefix hosted` for local development. The gateway uses Node 24 and the Railway Sandbox SDK.

Set `HELIX_PUBLIC_URL` to the service's HTTPS origin, `RAILWAY_TOKEN` to a production project token, and `HELIX_BASE_CHECKPOINT` to your prepared runtime checkpoint. Railway supplies `RAILWAY_ENVIRONMENT_ID` inside its service; set it explicitly when running the gateway elsewhere. Keep tokens in Railway's variables, never in Git.

Create the base checkpoint in the same Railway environment with `node hosted/prepare.mjs`. It builds the CPU image inside a fresh sandbox and saves `helix-base-v1`. Native Codex and Claude Code binaries come from Railway's sandbox image. The gateway copies the current runner into each new workspace. `node hosted/server.mjs` starts the gateway, using `PORT` or 3000. The public worker URLs require a random per-workspace bearer credential and never accept a browser session directly.

Google/GitHub sign-in buttons appear when their OAuth clients are configured. Set `HELIX_GITHUB_CLIENT_ID` and `HELIX_GITHUB_CLIENT_SECRET` for a GitHub OAuth app with callback `${HELIX_PUBLIC_URL}/account/oauth/github/callback`. For Google, set `HELIX_GOOGLE_CLIENT_ID` and `HELIX_GOOGLE_CLIENT_SECRET` for a Web application OAuth client with callback `${HELIX_PUBLIC_URL}/account/oauth/google/callback`; configure its consent screen for your intended users. Use basic profile access only. The app requests no repository write access, email inbox or Drive access.

Existing users can link these methods from **Connections** after signing in to their Helix account. Identities are matched by the provider's stable account ID, never by a matching name or email. Linking an identity already owned by another Helix account is rejected. Login uses one-time, browser-bound state and PKCE; provider access tokens are used to fetch identity, then discarded. Native Codex/Claude credentials remain separate. Password recovery and account deletion are not yet self-service. A single gateway owns workspace admission and idle saves; add shared admission control before scaling to multiple gateway instances.

## Run it locally

The ready-to-run ZIP includes the built interface. If your prerequisites and training runtime are already set up, extract it and run `npm start`; no dependency installation or interface build is needed.

The Mac preview ZIP contains **Helix ML.app**. Move it to Applications and open it to launch Helix in your browser. It connects automatically, so no terminal or pairing code is needed for ordinary use. Stop any older terminal runner first. **Connections → Quit Helix** stops the app and pauses an active experiment. The launcher uses `~/Library/Application Support/Helix ML` for data and logs, separately from a source checkout's `.helix` folder. Existing experiments are not moved automatically.

This is an unsigned launcher preview, not a prerequisite installer. Node, Python, Docker, the training image and agent sign-in must already be available. The launcher and browser flows are checked on Linux, but Finder launch and macOS approval still require physical Mac testing. The bundle follows Apple's [application bundle structure](https://developer.apple.com/library/archive/documentation/CoreFoundation/Conceptual/CFBundles/BundleTypes/BundleTypes.html).

Install Node.js 22.13+, Python 3.11+, Docker Desktop, and either the Codex or Claude Code CLI. Start Docker Desktop and sign in with `codex login` or `claude auth login`. Helix uses subscription authentication; it does not ask for API keys.

From this folder:

```sh
npm run local
```

This checks prerequisites, builds the CPU runtime if needed, installs the locked app dependencies, builds the interface, and starts Helix. For subsequent sessions, use `npm start`. After updating the source, run `npm run local` again. `npm run doctor` checks prerequisites without starting an experiment.

Open **http://127.0.0.1:4319**. The page connects automatically to the runner that serves it. Choose your agent at the bottom of the chat and attach your data with + or the file picker in Settings. Describe the experiment in the composer. Send immediately shows the submitted task with preparation progress; a failed startup keeps the message available to edit. The desktop setup panel shows the dataset, evaluation protocol and budget, with expandable model/strategy and deliverable controls. On smaller screens, open Settings from the composer. Draft settings are editable before connection. Keep Docker Desktop running, and keep the terminal open if you launched with npm. Closing the browser does not stop training. Ctrl+C pauses the terminal runner; after restarting it, the page reconnects and you can resume from saved history. A separately hosted UI still requires manual pairing under **Connections → Advanced connection**.

Before the first experiment, Helix checks that the selected agent can use its tools to write a training program, then runs that program on synthetic data and verifies prediction and model reload. A failed check blocks the experiment. It runs automatically when you send; there is no separate verification step. This uses your agent subscription, takes up to three minutes, and does not use your dataset or experiment budget. Successful checks are reused across runner restarts for up to seven days while the CLI version, Docker image, Node version/platform and Helix runner code remain unchanged. Passing this small CPU/Joblib check verifies the connection to the harness; it does not certify every model or framework.

Helix validates the dataset and requested splits before spending subscription usage on a new experiment. A second runner cannot open the same data folder while the first is alive. After an interrupted process, the next runner recovers its saved experiments. Use **Use these settings** on a saved experiment to create an editable draft with its data and configuration.

`npm run doctor` identifies missing prerequisites. On macOS, make sure Docker Desktop can share the project and dataset folders. Paths must not contain commas. To use a different Python executable, set `HELIX_PYTHON`. Experiment state and uploaded data live in `.helix/`; source datasets are never modified.

## First test

The bundled [measurements.csv](examples/measurements.csv) is synthetic test data, not a benchmark. Attach that file with **+ → Upload CSV**, enter `label` in **Settings → Target column**, and set the following:

| Setting | Value |
| --- | --- |
| Objective | Predict short or long from length and width |
| Target | `label` |
| Metric | Accuracy |
| Budget | 15 minutes, 3 trials |
| Model selection | Choose model |
| Model | `sklearn.tree.DecisionTreeClassifier` |
| Strategies | All off |
| Validation | Fixed holdout, seed 42 |
| Source | Python or notebook |
| Export | On, Joblib |

You can use **Check dataset** in Settings before submitting. Send your experiment with the arrow button or Enter. Shift+Enter adds a new line. This constrained configuration evaluates one fixed candidate; the trial count is a ceiling. Agent research and code review can take several minutes. Activity shows actual progress. After completion, click **Download solution** in the conversation. The ZIP contains source, an optional notebook, the fitted model and prediction entry point, configuration, row partitions, dependency versions, checksums, and measured results. Raw dataset rows are excluded.

For model search, allow several candidates and more time. CV multiplies training work by folds × seeds. All research, repairs, package installs, fits, and export checks share the time budget.

Candidates are researched and written one at a time. The first candidate is reviewed and evaluated before discovery of the next model begins. An explicitly selected estimator skips model discovery. Small candidates are supplied directly to the independent reviewer with their training schema, avoiding serial file reads. The interface shows the current trial's writing, review, repair or training stage and completed fold fits; a trial is counted as completed only after its selected validation protocol finishes. Independent code review, policy checks, scoring and model reload remain required.

A Linux CPU timing check with Codex, an 80-row synthetic CSV, a fixed DecisionTreeClassifier, one holdout and one seed reached its first measured score in 98.6 seconds after resume. Initial capability verification took 47.0 seconds; verification after restart took 0.6 seconds. Training, Joblib reload, reserved-test evaluation and download passed. These are one-run measurements; model discovery, repairs and larger validation protocols can still take several minutes.

## Inputs and evaluation

Supported tasks are supervised classification and regression from a labeled CSV. Text can be inline; image, audio and text files must be referenced by relative paths in explicitly declared asset columns under **Settings → Asset columns**. Only referenced assets are copied. This MVP does not infer labels from arbitrary media folders or implement detection, segmentation or generation.

Independent, grouped, and chronological splits are available. Repeated independent inputs are rejected; use a group column for repeated subjects. Set your test percentage in Settings, including 0% to omit the test partition. Fixed validation also has a configurable percentage of the training pool; CV supports 3, 5 or 10 folds. With 0% test, the selected model is refitted on all rows and exported without a test score. The **Seed** field takes an actual integer, such as `42`, not a count. Multiple actual values can be separated by commas, such as `42, 43`. Every requested seed/fold must complete before a candidate is compared. Scores and standard deviations are measured by the runner, outside training containers.

Results include a simple baseline measured on the same rows. Accuracy uses the most frequent training class, probability metrics use training class frequencies, RMSE uses the training mean, and MAE uses the training median. Each fold fits its baseline from its own training labels. Baselines also appear in the downloaded experiment record. A model can complete successfully without beating its baseline; the interface reports that outcome.

Local models train with two CPU cores, 3 GB RAM, no network, read-only inputs, and bounded outputs. Preparation limits are 128 MiB CSV, 100,000 rows, 200 columns, 1,000 classes, and 2 GiB referenced assets. One experiment runs at a time per runner. Pause interrupts the current candidate; resume restarts that candidate using the remaining budget. Final evaluation cannot be paused or retried after interruption, to preserve the one-test rule. Stop retains completed trials and artifacts.

## Exports and limits

Python and notebook output execute the same source. Notebooks run in a fresh Jupyter kernel during evaluation. Every requested model export is checked with its native format loader and a separate prediction process without training labels. Predictions must match exactly for labels or within `1e-6 × max(1, abs(value))` for numbers. Failed conversions/reloads reject the candidate; Helix never silently changes an explicit format.

Joblib and Pickle have passed complete native-agent acceptance runs. PyTorch, TorchScript, Keras, SavedModel and ONNX have loader/reload checks, but each requires an appropriate model and installed dependencies; their framework-specific end-to-end coverage remains future work. Choose **Native** when you do not need a specific format. Inspect generated code and load only artifacts from runs you trust.

Model restrictions and strategy permissions are included in research, tool availability, per-candidate plans, agent review, and syntax checks for known violations. Syntax checks cannot prove arbitrary Python compliance. Keep the generated plan/audit/source available for review. Dependency additions are isolated per candidate; the winner's versions and Docker image ID are recorded. Pretrained model revisions and licenses are recorded separately and must be restored for reproduction when referenced.

The hosted preview targets browsers on macOS, Windows and Linux. Its training workers are Linux CPU VMs. The optional local launcher still requires its documented prerequisites; native Windows execution is not supported by its POSIX process management. NVIDIA and Apple MPS acceleration and signed installers remain future work. Dataset schemas, objectives, generated code and tool results reach the selected agent provider through its native CLI. User credentials stay with that CLI; inherited user MCP servers and shell tools are disabled for these sessions.

## Development checks

`npm run package` builds ready-to-run source and Mac preview ZIPs in `output/`. They include the compiled interface, exclude experiment data and credentials, and contain file checksums. The build disables Rollup 4.64 tree shaking because its call-argument analysis stalls on this app's React imports; minification remains enabled.

```sh
npm test                 # API boundaries, preparation, metrics and policy checks
npm run test:hosted      # Account/session persistence, CSRF, OAuth state/PKCE, explicit account linking and user separation
npm run test:runtime     # Real CPU fits, notebooks, reloads and cancellation
npm run test:agents      # Native MCP write, real training, prediction and reload
npm run test:mvp         # Full Claude experiment, pause/restart, export and download
npm run test:acceptance  # Real UCI data, model search, refinement and portable exports
```

The agent checks consume subscription usage. To exercise Codex and Python/Pickle output:

```sh
HELIX_TEST_AGENT=codex HELIX_TEST_OUTPUT=py HELIX_TEST_FORMAT=pickle npm run test:mvp
```

For browser regressions, open the `npm run dev` URL with Playwright CLI, then run `playwright-cli run-code --filename local/ui_check.js`. It checks clearing numeric inputs, sending after login with stale status, and provider-specific errors using mocked account endpoints.

Runtime fixtures cover tabular, text, image and audio input handling; their synthetic scores do not establish real-world model quality. The UI can also run separately with `npm run dev` and `npm run bridge -- --origin http://127.0.0.1:5173`.

The opt-in acceptance check downloads checksum-pinned [Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing) data by Moro, Rita and Cortez, and [Bike Sharing](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset) data by Fanaee-T, both under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). It uses Codex for bank classification and Claude Code for bike regression, with a 30-minute experiment budget per dataset plus agent verification. Run one case with `npm run test:acceptance -- bank` or `-- bike`; `HELIX_TEST_AGENT` overrides the agent. Reports, provenance, source, and downloaded solutions stay in `output/acceptance/`.

Bank uses the supplied 4,521-row sample, omits post-call duration and represents unknown categories as missing cells. Client identifiers are unavailable, so random-split scores cannot rule out repeated-client leakage. Bike uses daily records, removes both target components and the row identifier, and evaluates later dates. Observed weather is an input, so the task is conditional demand estimation. The checks require multiple measured candidates, a measured refinement, six validation fits per candidate, improvement over the trivial baseline, and matching predictions from the downloaded model in isolation. Bank also pauses after a completed trial, restarts the runner, and resumes. These checks establish particular working paths; they do not guarantee improvement on new data.

The remaining macOS acceptance step needs physical Macs: install prerequisites in a clean environment, run `npm run local`, attach your own CSV, complete an experiment, and use the downloaded predictor. Also pause an experiment, close the runner, restart it, and resume. Record the Mac architecture, agent version and any failure. Automated checks on Linux cannot replace this step.

Recorded acceptance results, October 6, 2026, Linux/x64:

| Dataset / agent | Validation model / baseline | Test model / baseline |
| --- | --- | --- |
| Bank / Codex, AUROC | 0.7234 / 0.5000 | 0.7190 / 0.5000 |
| Bike / Claude Code, RMSE | 1335.15 / 2083.67 | 1130.61 / 2560.56 |

Both completed six trials, including measured refinements, and reproduced final predictions from the downloaded ZIP in a separate container. The bank run also passed pause/restart/resume after a completed trial. These are recorded runs, not promised scores for future agent-generated solutions. Prerequisite/setup, API/lifecycle, CPU runtime and desktop/mobile browser checks also passed. Physical Mac and independent friend tests are still outstanding.

An interrupted acceptance check preserves its experiments. Set `HELIX_ACCEPTANCE_ROOT` to its printed evidence folder and repeat the same command to resume within the saved time budget. A completed test evaluation is reused, never rerun for model selection.

See [HELIX_ML_SPEC.md](HELIX_ML_SPEC.md) for requirements and implementation history. [MIT licensed](LICENSE).
