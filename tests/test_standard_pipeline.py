import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from standard_pipeline import (
    WINDOWS_PATH_LIMIT,
    _resolve_npm_command,
    _unique_destination,
    decompress_packages,
    extract_archive_raw,
    resolve_dataset_archives,
    sha1_short,
)


class StandardPipelineTests(unittest.TestCase):
    def test_resolve_dataset_archives_reads_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "sample.tgz"
            archive.write_text("x", encoding="utf-8")
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps({"archives": [str(archive)]}),
                encoding="utf-8",
            )

            self.assertEqual(resolve_dataset_archives(manifest_path=manifest, dataset_dir=None), [archive])

    def test_unique_destination_shortens_windows_long_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dest_dir = root / ("nested" * 12) / ("materialized" * 8)
            dest_dir.mkdir(parents=True)
            src = root / ("DataDog__" + ("very-long-package-name-" * 8) + "1.0.0.zip")
            src.write_text("x", encoding="utf-8")

            destination = _unique_destination(dest_dir, src)

            self.assertLessEqual(len(str(destination)), WINDOWS_PATH_LIMIT)
            self.assertEqual(destination.suffix, ".zip")
            self.assertNotEqual(destination.name, "")

    def test_resolve_npm_command_prefers_windows_launcher(self) -> None:
        with patch("standard_pipeline.shutil.which") as which:
            which.side_effect = lambda name: {
                "npm.cmd": r"D:\\envs\\nodejs\\npm.cmd",
                "npm": r"D:\\envs\\nodejs\\npm.CMD",
            }.get(name)

            self.assertEqual(_resolve_npm_command(), r"D:\\envs\\nodejs\\npm.cmd")

    def test_extract_archive_raw_cleans_partial_directory_when_extraction_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "sample.tgz"
            archive.write_text("x", encoding="utf-8")
            temp_root = root / "out"
            temp_root.mkdir()

            def fail_extract(_: Path, dest: Path) -> None:
                (dest / "package").mkdir(parents=True, exist_ok=True)
                (dest / "package" / "partial.js").write_text("partial", encoding="utf-8")
                raise RuntimeError("simulated extraction failure")

            with patch("standard_pipeline.safe_extract_tar", side_effect=fail_extract):
                with self.assertRaisesRegex(RuntimeError, "simulated extraction failure"):
                    extract_archive_raw(archive, temp_root)

            self.assertEqual(list(temp_root.iterdir()), [])

    def test_decompress_packages_reports_failed_archives(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = root / "dataset"
            dataset_dir.mkdir()
            output_root = root / "work"
            good_archive = dataset_dir / "good.tgz"
            bad_archive = dataset_dir / "bad.tgz"
            good_archive.write_text("good", encoding="utf-8")
            bad_archive.write_text("bad", encoding="utf-8")

            def fake_extract(archive_path: Path, temp_dataset_path: Path) -> Path:
                if archive_path.name == "good.tgz":
                    extracted = temp_dataset_path / f"{archive_path.stem}_{sha1_short(archive_path)}"
                    (extracted / "package").mkdir(parents=True, exist_ok=True)
                    return extracted
                raise RuntimeError("blocked by antivirus")

            with patch("standard_pipeline.extract_archive_raw", side_effect=fake_extract):
                with self.assertRaisesRegex(RuntimeError, r"bad\.tgz.*blocked by antivirus"):
                    decompress_packages(dataset_dir, output_root)


if __name__ == "__main__":
    unittest.main()
