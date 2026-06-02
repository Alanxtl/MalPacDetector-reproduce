import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cli import CACHE_COMPLETE_MARKER, decompress_packages, extract_archive_raw, sha1_short


class CliDecompressionTests(unittest.TestCase):
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

            with patch("cli.safe_extract_tar", side_effect=fail_extract):
                with self.assertRaisesRegex(RuntimeError, "simulated extraction failure"):
                    extract_archive_raw(archive, temp_root)

            self.assertEqual(list(temp_root.iterdir()), [])

    def test_decompress_packages_reports_failed_archives(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = root / "dataset"
            dataset_dir.mkdir()
            output_root = root / ".decompressed-packages"
            output_root.mkdir()
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

            cwd = os.getcwd()
            os.chdir(root)
            try:
                with patch("cli.extract_archive_raw", side_effect=fake_extract):
                    with self.assertRaisesRegex(RuntimeError, r"bad\.tgz.*blocked by antivirus"):
                        decompress_packages(str(dataset_dir))
            finally:
                os.chdir(cwd)

    def test_decompress_packages_ignores_incomplete_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = root / "dataset"
            dataset_dir.mkdir()
            archive = dataset_dir / "sample.tgz"
            archive.write_text("sample", encoding="utf-8")
            cached_dir = root / ".decompressed-packages" / dataset_dir.name
            cached_dir.mkdir(parents=True)
            (cached_dir / "stale.txt").write_text("stale", encoding="utf-8")

            def fake_extract(archive_path: Path, temp_dataset_path: Path) -> Path:
                extracted = temp_dataset_path / f"{archive_path.stem}_{sha1_short(archive_path)}"
                (extracted / "package").mkdir(parents=True, exist_ok=True)
                return extracted

            cwd = os.getcwd()
            os.chdir(root)
            try:
                with patch("cli.extract_archive_raw", side_effect=fake_extract) as extract_mock:
                    result = decompress_packages(str(dataset_dir), use_cache=True)
            finally:
                os.chdir(cwd)

            self.assertEqual(Path(result), cached_dir.resolve())
            self.assertEqual(extract_mock.call_count, 1)
            self.assertFalse((cached_dir / "stale.txt").exists())
            self.assertTrue((cached_dir / CACHE_COMPLETE_MARKER).exists())


if __name__ == "__main__":
    unittest.main()
