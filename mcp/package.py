"""Build runtime-inclusive downloads. Python is needed by the builder, not the user."""
import hashlib
import json
import platform
import shutil
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output" / "mcp"
NODE = "24.21.0"
UV = "0.13.0"
UV_LICENSES = {
    "LICENSE-MIT": "860e3d7a86b84e6a7012c7a635fc64df475cebc6cce34dfeb73a5982ec58176c",
    "LICENSE-APACHE": "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4",
}
# Pin upstream archives; never package an unverified executable.
TARGETS = {
    "macos-arm64": ("darwin", "arm64", "aarch64-apple-darwin", "bed7eea5325e1108f32ce5228ddd6a5f0f08a499ee42aa7442aea583702f6057", "a9c1b29002cf3c83f07fa9cd8a887a3be0107d90e23189721221e7257db8e3d6"),
    "macos-x64": ("darwin", "x64", "x86_64-apple-darwin", "1462cb3b3046b815cf8ea436d3da450ec1a9f11dac7e5a46b0ada5305d7e8097", "5f44dcbde809b632f47c36fadb241cb4d6f9af71d0c8f172b5d2026d3dde742c"),
    "windows-x64": ("win32", "x64", "x86_64-pc-windows-msvc", "158f7685b44de51f6c0df1d153526cbcd3e1bc739a8dfc607721cef75de9e541", "088962f9e7b7bd9ea740c04c650b2a21c8928c345bd99ac24350dc924dba656c"),
    "linux-x64": ("linux", "x64", "x86_64-unknown-linux-gnu", "6e1db87ef58b8819e5d5402eff1536491b18edd8eb7bee5ef7897876e88dc5ff", "1468ebd5a5541121837c5a2817b9972ba6090fa6caa3d142620850a47fb75154"),
}


def download(url, checksum):
    cache = OUT / ".downloads" / url.rsplit("/", 1)[-1]
    cache.parent.mkdir(parents=True, exist_ok=True)
    if not cache.exists():
        print(f"Downloading {cache.name}…", flush=True)
        with urllib.request.urlopen(url, timeout=120) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != checksum:
            raise ValueError(f"Checksum mismatch: {url}")
        cache.write_bytes(data)
    if hashlib.sha256(cache.read_bytes()).hexdigest() != checksum:
        raise ValueError(f"Cached download failed integrity check: {cache}")
    return cache


def member(archive, name):
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as source:
            matches = [p for p in source.namelist() if p == name or p.endswith("/" + name)]
            assert len(matches) == 1, (archive, name, matches)
            return source.read(matches[0])
    with tarfile.open(archive) as source:
        matches = [p for p in source.getmembers() if p.isfile() and (p.name == name or p.name.endswith("/" + name))]
        assert len(matches) == 1, (archive, name, matches)
        return source.extractfile(matches[0]).read()


def build(target):
    system, arch, uv_target, node_hash, uv_hash = TARGETS[target]
    windows = system == "win32"
    suffix = ".zip" if windows else ".tar.gz"
    node_target = f"{'win' if windows else system}-{arch}"
    node_archive = download(f"https://nodejs.org/dist/v{NODE}/node-v{NODE}-{node_target}{suffix}", node_hash)
    uv_archive = download(f"https://github.com/astral-sh/uv/releases/download/{UV}/uv-{uv_target}{suffix}", uv_hash)
    bundle = OUT / target / "Helix ML"
    if bundle.exists():
        shutil.rmtree(bundle)
    runtime = bundle / "runtime"
    runtime.mkdir(parents=True)
    node_name, uv_name = ("node.exe", "uv.exe") if windows else ("node", "uv")
    (runtime / node_name).write_bytes(member(node_archive, node_name if windows else "bin/node"))
    (runtime / uv_name).write_bytes(member(uv_archive, uv_name))
    (runtime / "NODE-LICENSE.txt").write_bytes(member(node_archive, f"node-v{NODE}-{node_target}/LICENSE"))
    for name, checksum in UV_LICENSES.items():
        shutil.copyfile(download(f"https://raw.githubusercontent.com/astral-sh/uv/{UV}/{name}", checksum), runtime / f"UV-{name}.txt")
    for name in [node_name, uv_name]:
        (runtime / name).chmod(0o755)
    (runtime / "manifest.json").write_text(json.dumps({"platform": system, "arch": arch, "nodeVersion": NODE, "uvVersion": UV, "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(runtime.iterdir())}}, indent=2) + "\n")
    for directory in ["local", "mcp", "examples"]:
        for source in (ROOT / directory).rglob("*"):
            if source.is_file() and source.suffix in {".mjs", ".py", ".json", ".md", ".txt", ".csv"} and "__pycache__" not in source.parts:
                destination = bundle / source.relative_to(ROOT)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
    shutil.copyfile(ROOT / "LICENSE", bundle / "LICENSE")
    if windows:
        (bundle / "Install Helix.cmd").write_bytes(b'@echo off\r\n"%~dp0runtime\\node.exe" "%~dp0mcp\\installer.mjs"\r\nset "HELIX_INSTALL_EXIT=%errorlevel%"\r\necho.\r\npause\r\nexit /b %HELIX_INSTALL_EXIT%\r\n')
    else:
        launcher = bundle / ("Install Helix.command" if system == "darwin" else "Install Helix.sh")
        launcher.write_text('#!/bin/bash\ncd -- "$(dirname -- "$0")" || exit 1\n./runtime/node ./mcp/installer.mjs\nhelix_install_exit=$?\nprintf "\\nPress Enter to close…"\nread -r helix_close\nexit "$helix_install_exit"\n')
        launcher.chmod(0o755)
    (bundle / "READ ME.txt").write_text("Helix ML — local MCP\n\n1. Extract this ZIP completely.\n2. Open Install Helix and choose Codex or Claude Code.\n3. Restart your agent session, then ask it to use Helix on your CSV.\n\nYour chosen agent CLI must already be installed and signed in.\nNode is bundled. Private Python and ML packages download automatically\non first setup; allow a few minutes and keep the installer open.\nNo Docker, administrator access, global runtime install or Helix login.\n\nFiles and experiments: ~/.helix-ml/mcp\nAfter setup, you can delete this downloaded folder.\nTo update or repair, open the installer again. Experiments are kept.\n\nGenerated Python runs with your user permissions, including file/network\naccess. This is local execution, not an OS sandbox.\n\nThese preview installers are unsigned. Your OS may request approval.\nHelp: https://github.com/kavinravi/Helix-ML#use-your-computer-through-mcp--no-docker\n")
    archive = OUT / f"helix-mcp-{target}.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as result:
        for source in sorted(bundle.rglob("*")):
            if source.is_file():
                result.write(source, source.relative_to(bundle.parent))
    print(f"Built {archive} ({archive.stat().st_size / 1024**2:.1f} MiB)", flush=True)


if __name__ == "__main__":
    default = {"Darwin": "macos", "Windows": "windows", "Linux": "linux"}[platform.system()]
    default += "-arm64" if platform.machine().lower() in {"arm64", "aarch64"} else "-x64"
    for target in list(TARGETS) if sys.argv[1:] == ["--all"] else sys.argv[1:] or [default]:
        if target not in TARGETS:
            raise SystemExit(f"Supported downloads: {', '.join(TARGETS)}")
        build(target)
