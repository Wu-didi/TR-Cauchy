"""Build a clean, deterministic TR-Cauchy paper-code ZIP archive."""

from __future__ import annotations

import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Optional
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "dist" / "TR_Cauchy_Code.zip"
ARCHIVE_ROOT = Path("TR-Cauchy")

ROOT_FILES = (
    ".gitignore",
    "LICENSE",
    "README.md",
    "PAPER_CODE_ATTACHMENT.md",
    "MANIFEST.in",
    "requirements.txt",
    "requirements-tr-cauchy.txt",
    "requirements-experiments.txt",
    "requirements-dev.txt",
    "pyproject.toml",
    "setup.py",
    "famo.py",
    "SOURCE_PROVENANCE.json",
    "VALIDATION.md",
)

SOURCE_PATTERNS = (
    "requirements*.txt",
    "docs/**/*.md",
    "LICENSES/*.txt",
    "examples/**/*.py",
    "examples/**/*.md",
    ".github/**/*.yml",
    ".github/**/*.yaml",
    ".github/**/*.md",
    "methods/*.py",
    "experiments/**/*.py",
    "experiments/**/README.md",
    "experiments/**/run.sh",
    "tests/*.py",
    "scripts/build_paper_attachment.py",
)

OPTIONAL_ROOT_FILES = (
    "CONTRIBUTING.md", "SECURITY.md", "CODE_OF_CONDUCT.md", "CITATION.cff",
    "CHANGELOG.md", "THIRD_PARTY_NOTICES.md",
)

# Source globs must not sweep in locally generated data, tools, or environment files.
EXCLUDED_DIRECTORIES = frozenset(
    {
        ".git", ".agents", ".codex", ".venv", "venv", "env", "node_modules",
        "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
        "data", "dataset", "datasets", "wandb", "dist", "build",
        "logs", "log", "trainlogs", "outputs", "output", "save", "saved_models",
        "checkpoints", "resource_benchmarks",
    }
)


def _safe_source(path: Path, root: Path) -> bool:
    relative = path.relative_to(root)
    if any(
        part in EXCLUDED_DIRECTORIES
        or (part.startswith(".") and not (index == 0 and part == ".github"))
        for index, part in enumerate(relative.parts[:-1])
    ):
        return False
    # Never follow linked files/directories into another checkout or private data.
    return not any(root.joinpath(*relative.parts[:index]).is_symlink() for index in range(1, len(relative.parts) + 1))


def selected_files(root: Optional[Path] = None) -> list[Path]:
    root = ROOT if root is None else Path(root).resolve()
    files = {root / name for name in ROOT_FILES}
    missing = sorted(path for path in files if not path.is_file())
    if missing:
        raise FileNotFoundError(f"missing attachment files: {missing}")
    unsafe = sorted(path for path in files if not _safe_source(path, root))
    if unsafe:
        raise ValueError(f"linked attachment inputs are not supported: {unsafe}")
    files.update(root / name for name in OPTIONAL_ROOT_FILES if (root / name).is_file())
    for pattern in SOURCE_PATTERNS:
        files.update(path for path in root.glob(pattern) if path.is_file() and _safe_source(path, root))
    return sorted((path for path in files if _safe_source(path, root)), key=lambda path: path.relative_to(root).as_posix())


def build_archive(root: Optional[Path] = None, output: Optional[Path] = None) -> Path:
    root = ROOT if root is None else Path(root).resolve()
    output = OUTPUT if output is None else Path(output)
    # Resolve and validate required inputs before touching an existing release ZIP.
    files = selected_files(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent, delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        with ZipFile(temporary_path, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
            for path in files:
                relative = path.relative_to(root)
                info = ZipInfo((ARCHIVE_ROOT / relative).as_posix())
                # Fixed metadata makes repeated builds byte-for-byte reproducible.
                info.date_time = (2026, 1, 1, 0, 0, 0)
                info.compress_type = ZIP_DEFLATED
                info.external_attr = (0o100755 if path.name == "run.sh" else 0o100644) << 16
                archive.writestr(info, path.read_bytes())
        os.replace(temporary_path, output)
    finally:
        temporary_path.unlink(missing_ok=True)
    return output


if __name__ == "__main__":
    output = build_archive()
    print(output.relative_to(ROOT))
