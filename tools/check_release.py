"""Fail release builds that omit licensing or the corresponding source."""
import hashlib
import io
import sys
import tarfile
import zipfile
from email.parser import BytesParser
from pathlib import Path

from prepare_release_sources import GSL_ARCHIVE, GSL_SHA256


def check_bundle(content):
    with tarfile.open(fileobj=io.BytesIO(content), mode="r:gz") as bundle:
        names = bundle.getnames()
        root = names[0].split("/")[0]
        required = ["LICENSE", "COPYING", "THIRD_PARTY_NOTICES", "pyproject.toml", "setup.py",
                    "MANIFEST.in", "src/gaiamock/gaiamock.py", "src/gaiamock/kepler_solve_astrometry.c",
                    "tools/build_gsl.sh", "tools/prepare_release_sources.py", "tools/check_release.py",
                    ".github/workflows/wheels.yml", "vendor/" + GSL_ARCHIVE]
        for path in required:
            assert root + "/" + path in names, "Missing corresponding source: " + path
        source = bundle.extractfile(root + "/vendor/" + GSL_ARCHIVE).read()
        assert hashlib.sha256(source).hexdigest() == GSL_SHA256, "Incorrect GSL source"


def check(path):
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            metadata_name = next(n for n in names if n.endswith(".dist-info/METADATA"))
            metadata = BytesParser().parsebytes(archive.read(metadata_name))
            prefix = metadata_name.rsplit("/", 1)[0] + "/licenses/"
            for name in ("LICENSE", "COPYING", "THIRD_PARTY_NOTICES"):
                assert prefix + name in names, "Missing license: " + name
            assert any("/libgsl" in n and "libgslcblas" not in n for n in names), "GSL not bundled"
            assert any("/libgslcblas" in n for n in names), "GSL CBLAS not bundled"
            source = [n for n in names if n.startswith("gaiamock/_sources/") and n.endswith("-source.tar.gz")]
            assert len(source) == 1, "Expected exactly one corresponding-source archive"
            check_bundle(archive.read(source[0]))
    else:
        with tarfile.open(path, "r:gz") as archive:
            names = archive.getnames()
            root = names[0].split("/")[0]
            metadata = BytesParser().parsebytes(archive.extractfile(root + "/PKG-INFO").read())
            for name in ("LICENSE", "COPYING", "THIRD_PARTY_NOTICES", "vendor/" + GSL_ARCHIVE):
                assert root + "/" + name in names, "Missing source distribution file: " + name
            source = [n for n in names if "/src/gaiamock/_sources/" in n and n.endswith("-source.tar.gz")]
            assert len(source) == 1, "Expected exactly one corresponding-source archive"
            check_bundle(archive.extractfile(source[0]).read())
    assert metadata["License-Expression"] == "GPL-3.0-or-later", "Incorrect release license"
    print("PASS:", path.name, "contains GPL/MIT notices and the pinned corresponding source")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: check_release.py <wheel-or-sdist> [...]")
    for filename in sys.argv[1:]:
        check(Path(filename))
