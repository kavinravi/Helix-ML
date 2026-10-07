"""Bundle public artifacts, with checksums, without copying user data."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile

folder = Path(sys.argv[1])
paths = sorted(p for p in folder.rglob("*") if p.name not in {"helix-solution.zip", "checksums.json"})
if any(p.is_symlink() for p in paths):
    raise ValueError("Artifact symlinks are forbidden")
files = [p for p in paths if p.is_file()]
checksums = {}
for path in files:
    with path.open("rb") as stream:
        checksums[str(path.relative_to(folder))] = hashlib.file_digest(stream, "sha256").hexdigest()
checksum_file = folder / "checksums.json"
checksum_file.write_text(json.dumps(checksums, indent=2))
with zipfile.ZipFile(folder / "helix-solution.zip", "w", zipfile.ZIP_DEFLATED) as archive:
    for path in files + [checksum_file]:
        archive.write(path, path.relative_to(folder))
