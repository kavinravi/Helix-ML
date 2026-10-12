import { fileURLToPath } from "node:url";
import { runProcess } from "../local/process.mjs";

export const TORCH_VERSION = "2.14.1";
const probeScript = fileURLToPath(new URL("../local/compute_probe.py", import.meta.url));
export async function probeCompute(python, hardwareOnly = false) {
  return JSON.parse((await runProcess(python, ["-I", probeScript, ...(hardwareOnly ? ["--hardware"] : [])], { timeout: 60_000 })).output);
}

export function torchPlan(hardware) {
  if (hardware.platform === "darwin" && hardware.architecture === "arm64") {
    if (Number(hardware.macos.split(".")[0]) < 14) return { reason: "GPU setup needs macOS 14 or newer. CPU training is available." };
    return { device: "mps", index: null };
  }
  const gpu = hardware.gpus[0];
  if (!gpu) return { device: "cpu" };
  // CUDA 13 drops older architectures; keep the legacy CUDA 12.6 wheel for them.
  const driver = gpu.driver.split(".").map(Number);
  const minimumMinor = hardware.platform === "win32" ? 88 : 65;
  const minimumPatch = hardware.platform === "win32" ? 0 : 6;
  const modernDriver = driver[0] > 580 || driver[0] === 580 && (driver[1] > minimumMinor || driver[1] === minimumMinor && (driver[2] || 0) >= minimumPatch);
  if (modernDriver && gpu.capability >= 7.5) return { device: "cuda", index: "https://download.pytorch.org/whl/cu130" };
  if (driver[0] >= 560 && gpu.capability >= 5 && gpu.capability < 10) return { device: "cuda", index: "https://download.pytorch.org/whl/cu126" };
  return { reason: "This GPU/driver combination is not supported by Helix's PyTorch wheels. Update the NVIDIA driver. CPU training is available." };
}

export function selectDevice(compute, requested = "auto") {
  if (!["auto", "cpu", "cuda", "mps"].includes(requested)) throw new Error("Choose auto, cpu, cuda or mps.");
  const available = compute?.available || ["cpu"];
  if (requested === "auto") return ["cuda", "mps", "cpu"].find(device => available.includes(device));
  if (!available.includes(requested)) throw new Error(`${requested.toUpperCase()} is unavailable. ${compute?.notes?.join(" ") || "Reopen the Helix installer to configure GPU support, or choose CPU."}`);
  return requested;
}

export function trialDevice(compute, environment, requested) {
  const device = requested || environment.device || "cpu";
  if (environment.requestedDevice && environment.requestedDevice !== "auto" && device !== environment.requestedDevice)
    throw new Error(`This experiment requires ${environment.requestedDevice.toUpperCase()}. Create a new experiment to change that requirement.`);
  return selectDevice(compute, device);
}

export async function setupCompute(runtime, report = console.error) {
  const hardware = await probeCompute(runtime.python, true);
  const plan = torchPlan(hardware);
  let compute = await probeCompute(runtime.python);
  if (plan.reason) compute.notes.push(plan.reason);
  if (["cuda", "mps"].includes(plan.device) && !compute.available.includes(plan.device)) {
    report(`Setting up ${plan.device.toUpperCase()} support. The first GPU download can be several GB…`);
    const progress = setInterval(() => report("GPU dependencies are still downloading/installing…"), 15_000);
    try {
      await runProcess(runtime.python, ["-m", "pip", "install", "--disable-pip-version-check", "--no-input", "--upgrade", `torch==${TORCH_VERSION}`, ...(plan.index ? ["--index-url", plan.index] : [])], { timeout: 1_800_000, onLine: report });
      compute = await probeCompute(runtime.python);
    } catch (error) {
      compute = await probeCompute(runtime.python);
      compute.notes.push("GPU dependency setup failed: " + error.message.slice(-1000) + ". Reopen the installer to retry.");
    } finally { clearInterval(progress); }
  }
  report(`Compute ready: ${selectDevice(compute).toUpperCase()}${compute.cuda ? ` (${compute.cuda.name})` : ""}.`);
  for (const note of compute.notes) report(note);
  return compute;
}
