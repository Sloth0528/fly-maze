"""Package only Git-indexed project files, including the two bundled v2.1 brains.

Run from a checkout: python tools/build_release.py
No virtual environment, credentials, raw downloads or untracked experiments are packaged.
"""

import hashlib
import json
from pathlib import Path
import subprocess
import zipfile


def build(output=None):
    root = Path(__file__).resolve().parents[1]
    version = (root/"VERSION").read_text(encoding="utf-8").strip()
    output = root/"dist"/f"fly-maze-v{version}.zip" if output is None else Path(output)
    if output.exists():
        raise FileExistsError(output)
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=root).decode("utf-8").split("\0")
    names = sorted(name for name in tracked if name)
    required = {"START_V2_1.cmd", "START_V2_1.command", "RUNNING.md", "run_v2_1.py",
                "models/connectome_v2_1.pt", "models/gru_v2_1.pt", "data/flywire_cx_512.npz",
                "data/flywire_cx_512.json", "data/DATA_LICENSE.md"}
    if not required.issubset(names):
        raise ValueError(f"Git index is missing release files: {sorted(required-set(names))}")
    for name in names:
        resolved = (root/name).resolve()
        if not resolved.is_relative_to(root) or any(part in (".venv", ".git", "work", "raw") for part in Path(name).parts):
            raise ValueError(f"unexpected release path: {name}")
    manifest = {"version": version, "files": {name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in names}}
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in names:
            info = zipfile.ZipInfo.from_file(root/name, arcname=name)
            info.create_system = 3
            executable = name.endswith((".command", ".sh"))
            info.external_attr = (0o100755 if executable else 0o100644) << 16
            archive.writestr(info, (root/name).read_bytes(), compress_type=zipfile.ZIP_DEFLATED)
        archive.writestr("RELEASE_MANIFEST.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise ValueError("archive CRC failed")
        for name, digest in manifest["files"].items():
            if hashlib.sha256(archive.read(name)).hexdigest() != digest:
                raise ValueError(f"archive digest failed: {name}")
        assert archive.getinfo("START_V2_1.command").external_attr >> 16 == 0o100755
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    checksum = output.with_suffix(output.suffix+".sha256")
    with checksum.open("x", encoding="utf-8") as handle:
        handle.write(f"{digest}  {output.name}\n")
    print(json.dumps({"archive": str(output), "files": len(names)+1, "sha256": digest, "bytes": output.stat().st_size}, indent=2))
    return output


if __name__ == "__main__":
    build()
