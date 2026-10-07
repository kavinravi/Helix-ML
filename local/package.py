"""Package the built app; exclude experiments, credentials and installed dependencies."""
import hashlib
import json
from pathlib import Path
import plistlib
import shutil
import subprocess
import tempfile
import zipfile

project = Path(__file__).resolve().parent.parent
output = project / "output"
output.mkdir(exist_ok=True)
if not (project / "dist/index.html").is_file():
    raise SystemExit("Run npm run build before packaging.")

with tempfile.TemporaryDirectory(prefix="helix-package-") as temp:
    staging = Path(temp)
    app = staging / "Helix ML.app/Contents"
    source = app / "Resources/helix"
    source.mkdir(parents=True)
    for name in [".gitignore", "LICENSE", "README.md", "HELIX_ML_SPEC.md", "package.json", "package-lock.json", "tsconfig.json", "vite.config.mjs", "index.html"]:
        shutil.copy2(project / name, source / name)
    for name in ["src", "public", "local", "examples", "dist"]:
        shutil.copytree(project / name, source / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    subprocess.run(["python3", str(project / "local/bundle.py"), str(source)], check=True)
    shutil.move(source / "helix-solution.zip", output / "helix-ml.zip")
    (app / "MacOS").mkdir()
    launcher = app / "MacOS/Helix"
    launcher.write_text('''#!/bin/zsh
set -eu
umask 077
export PATH="$HOME/.local/bin:$HOME/.volta/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"
if ! command -v node >/dev/null 2>&1 && [ -f "$HOME/.nvm/nvm.sh" ]; then
  set +u
  . "$HOME/.nvm/nvm.sh"
  set -u
fi
if ! command -v node >/dev/null 2>&1; then
  /usr/bin/osascript -e 'display dialog "Install Node.js 22.13 or newer, then reopen Helix." with title "Helix ML" buttons {"OK"} default button "OK"'
  exit 1
fi
export HELIX_DATA_DIR="$HOME/Library/Application Support/Helix ML"
mkdir -p "$HELIX_DATA_DIR"
app_root="$(cd "$(dirname "$0")/../Resources/helix" && pwd)"
exec node "$app_root/local/desktop.mjs" >> "$HELIX_DATA_DIR/launcher.log" 2>&1
''')
    launcher.chmod(0o755)
    with (app / "Info.plist").open("wb") as stream:
        plistlib.dump({
            "CFBundleName": "Helix ML", "CFBundleDisplayName": "Helix ML",
            "CFBundleIdentifier": "com.kavinravi.helixml", "CFBundleExecutable": "Helix",
            "CFBundlePackageType": "APPL", "CFBundleVersion": "1",
            "CFBundleShortVersionString": "0.1.0", "LSUIElement": True,
        }, stream)
    (staging / "READ ME.txt").write_text(
        "Move Helix ML.app to Applications and open it. The interface opens in your browser and connects automatically.\n"
        "Prerequisites: Node.js 22.13+, Python 3.11+, running Docker Desktop with the Helix CPU image, and a signed-in Codex or Claude CLI.\n"
        "This launcher does not install prerequisites. If you already ran npm run local, they should be ready.\n"
        "Stop an older terminal runner before opening this version. Use Connections > Quit Helix to stop it.\n"
        "Uploads and experiments: ~/Library/Application Support/Helix ML. This is separate from a source checkout's .helix folder.\n"
        "This preview is unsigned and has not been physically tested on macOS. macOS may require approval to open it.\n"
    )
    paths = sorted(p for p in staging.rglob("*") if p.is_file())
    if any(p.is_symlink() for p in staging.rglob("*")):
        raise ValueError("Do not package symlinks")
    checksums = {str(p.relative_to(staging)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    checksum_file = staging / "checksums.json"
    checksum_file.write_text(json.dumps(checksums, indent=2))
    with zipfile.ZipFile(output / "helix-ml-macos.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in paths + [checksum_file]:
            archive.write(path, path.relative_to(staging))
    with zipfile.ZipFile(output / "helix-ml-macos.zip") as archive:
        assert archive.testzip() is None
        assert archive.getinfo("Helix ML.app/Contents/MacOS/Helix").external_attr >> 16 & 0o111
        for name, digest in checksums.items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == digest, name
print("Created output/helix-ml.zip and output/helix-ml-macos.zip")
