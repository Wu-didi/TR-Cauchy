"""Release packaging checks; these require only the Python standard library."""

import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from zipfile import ZipFile


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_paper_attachment.py"
SPEC = importlib.util.spec_from_file_location("build_paper_attachment", SCRIPT)
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


class PackagingSafetyTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.output = self.root / "dist" / "release.zip"
        self.output.parent.mkdir()
        self.output.write_bytes(b"previous release")
        for name in builder.ROOT_FILES:
            path = self.root / name
            path.write_text(f"fixture: {name}\n", encoding="utf-8")

    def test_missing_input_preserves_previous_release(self):
        (self.root / "LICENSE").unlink()
        with self.assertRaises(FileNotFoundError):
            builder.build_archive(self.root, self.output)
        self.assertEqual(self.output.read_bytes(), b"previous release")
        self.assertEqual(list(self.output.parent.iterdir()), [self.output])

    def test_write_failure_preserves_previous_release_and_removes_temporary_file(self):
        with patch.object(builder.ZipFile, "writestr", side_effect=OSError("simulated disk failure")):
            with self.assertRaises(OSError):
                builder.build_archive(self.root, self.output)
        self.assertEqual(self.output.read_bytes(), b"previous release")
        self.assertEqual(list(self.output.parent.iterdir()), [self.output])

    def test_archive_is_repeatable_and_excludes_generated_or_linked_files(self):
        included = (
            "methods/example.py",
            "experiments/toy/run.sh",
            "requirements-dev.txt",
            "docs/USAGE.md",
            "LICENSES/upstream.txt",
            "CHANGELOG.md",
            "THIRD_PARTY_NOTICES.md",
            "examples/train_synthetic.py",
            ".github/workflows/ci.yml",
            "SOURCE_PROVENANCE.json",
            "VALIDATION.md",
        )
        excluded = (
            "experiments/toy/dataset/private.py",
            "experiments/toy/.venv/private.py",
            "experiments/toy/trainlogs/private.py",
            "experiments/toy/__pycache__/private.py",
            "experiments/toy/private.npy",
            ".github/.private/secret.yml",
        )
        for name in included + excluded:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("fixture\n", encoding="utf-8")
        (self.root / "methods" / "linked.py").symlink_to(self.root / "LICENSE")
        builder.build_archive(self.root, self.output)
        first_build = self.output.read_bytes()
        builder.build_archive(self.root, self.output)
        self.assertEqual(self.output.read_bytes(), first_build)
        with ZipFile(self.output) as archive:
            names = set(archive.namelist())
            prefix = builder.ARCHIVE_ROOT.as_posix() + "/"
            self.assertTrue(all(prefix + name in names for name in included))
            self.assertFalse(any(prefix + name in names for name in excluded))
            self.assertNotIn(prefix + "methods/linked.py", names)
            self.assertEqual(archive.testzip(), None)

    def test_required_symlink_is_rejected_before_replacing_archive(self):
        license_path = self.root / "LICENSE"
        license_path.unlink()
        license_path.symlink_to(self.root / "README.md")
        with self.assertRaises(ValueError):
            builder.build_archive(self.root, self.output)
        self.assertEqual(self.output.read_bytes(), b"previous release")


if __name__ == "__main__":
    unittest.main()
