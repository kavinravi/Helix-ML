"""Inspect accelerators and verify a small forward/backward pass before advertising them."""
import csv
import json
import platform
import subprocess
import sys


def probe(hardware_only=False):
    result = {"available": ["cpu"], "gpus": [], "notes": [], "platform": sys.platform,
              "architecture": platform.machine().lower(), "macos": platform.mac_ver()[0]}
    try:
        output = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,compute_cap", "--format=csv,noheader,nounits"],
                                capture_output=True, text=True, timeout=10, check=True).stdout
        for row in csv.reader(output.splitlines(), skipinitialspace=True):
            result["gpus"].append({"name": row[0], "driver": row[1], "memoryMiB": float(row[2]), "capability": float(row[3])})
    except FileNotFoundError:
        pass
    except (subprocess.SubprocessError, ValueError, IndexError) as error:
        result["notes"].append("NVIDIA detection failed: " + str(getattr(error, "stderr", None) or error)[:500])
    if hardware_only:
        return result
    try:
        import torch
    except ModuleNotFoundError as error:
        if error.name != "torch":
            result["notes"].append("PyTorch dependency is missing: " + str(error)[:500])
        return result
    except Exception as error:
        result["notes"].append("PyTorch could not load: " + str(error)[:500])
        return result
    result["torchVersion"] = torch.__version__
    result["torchIndex"] = ("https://download.pytorch.org/whl/cu" + torch.version.cuda.replace(".", "")) if torch.version.cuda else (None if sys.platform == "darwin" else "https://download.pytorch.org/whl/cpu")
    for device in ["cuda", "mps"]:
        try:
            available = torch.cuda.is_available() if device == "cuda" else torch.backends.mps.is_available()
            if not available:
                continue
            x = torch.ones((8, 8), device=device, requires_grad=True)
            (x @ x).sum().backward()
            assert float(x.grad[0, 0].cpu()) == 16.0
            if device == "cuda":
                torch.cuda.synchronize()
                properties = torch.cuda.get_device_properties(0)
                result["cuda"] = {"name": properties.name, "memoryMiB": properties.total_memory // 1024**2,
                                  "capability": f"{properties.major}.{properties.minor}", "runtime": torch.version.cuda}
            else:
                torch.mps.synchronize()
                result["mps"] = {"name": "Apple Metal GPU", "memoryMiB": torch.mps.recommended_max_memory() // 1024**2}
            result["available"].append(device)
            del x
        except Exception as error:
            result["notes"].append(f"{device.upper()} check failed: {str(error)[:500]}")
    if result["gpus"] and "cuda" not in result["available"]:
        result["notes"].append("An NVIDIA GPU was detected but CUDA training is unavailable. Update the NVIDIA driver and reopen the Helix installer.")
    if sys.platform == "darwin" and platform.machine().lower() == "arm64" and "mps" not in result["available"]:
        result["notes"].append("MPS is unavailable. This runtime needs Apple Silicon, macOS 14+, and an accessible Metal GPU.")
    return result


if __name__ == "__main__":
    print(json.dumps(probe("--hardware" in sys.argv)))
