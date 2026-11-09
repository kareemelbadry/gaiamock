"""Prepare the exact GSL source and corresponding-source bundle for a release."""
import hashlib
import re
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GSL_ARCHIVE = "gsl-2.8.tar.gz"
GSL_URL = "https://ftp.gnu.org/gnu/gsl/" + GSL_ARCHIVE
GSL_SHA256 = "6a99eeed15632c6354895b1dd542ed5a855c0f15d9ad1326c6fe2b2c9e423190"


def main():
    archive = ROOT / "vendor" / GSL_ARCHIVE
    archive.parent.mkdir(exist_ok=True)
    if not archive.exists():
        print("Downloading", GSL_URL, flush=True)
        with urllib.request.urlopen(GSL_URL, timeout=120) as response:
            content = response.read()
        if hashlib.sha256(content).hexdigest() != GSL_SHA256:
            raise RuntimeError("GSL source checksum does not match the pinned release")
        archive.write_bytes(content)
    if hashlib.sha256(archive.read_bytes()).hexdigest() != GSL_SHA256:
        raise RuntimeError("GSL source checksum does not match the pinned release")

    version = re.search(r'^version = "([^"]+)"$',
                        (ROOT / "pyproject.toml").read_text(), re.MULTILINE).group(1)
    source_name = f"gaiamock-{version}-source"
    source_dir = ROOT / "src" / "gaiamock" / "_sources"
    source_dir.mkdir(exist_ok=True)
    paths = [ROOT / name for name in (
        "LICENSE", "COPYING", "THIRD_PARTY_NOTICES", "README.md",
        "pyproject.toml", "setup.py", "MANIFEST.in", ".github/workflows/wheels.yml")]
    for pattern in ("src/gaiamock/*.py", "src/gaiamock/*.c", "tests/*.py", "tools/*.py", "tools/*.sh"):
        paths.extend(ROOT.glob(pattern))
    paths.append(archive)
    output = source_dir / (source_name + ".tar.gz")
    with tarfile.open(output, "w:gz") as bundle:
        for path in sorted(paths):
            bundle.add(path, arcname=str(Path(source_name) / path.relative_to(ROOT)))
    print("Prepared", output)


if __name__ == "__main__":
    main()
