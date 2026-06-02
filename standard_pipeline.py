from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import stat
import subprocess
import tarfile
import time
import zipfile
from pathlib import Path

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from conf import ROOT_PATH
from training import (
    ActionEnum,
    ModelEnum,
    PreprocessMethodEnum,
    predict_package_MLP,
    predict_package_NB,
    predict_package_RF,
    predict_package_SVM,
    train,
)
from training.src.commons import (
    MLP_path,
    classifier_save_path,
    mlp_scaler_save_path,
    nb_path,
    nb_scaler_save_path,
    rf_classifier_path,
    rf_scaler_save_path,
    svm_path,
    svm_scaler_save_path,
)


ARCHIVE_EXTS = (".tgz", ".tar.gz", ".tar", ".zip")
WINDOWS_PATH_LIMIT = 259


def load_manifest_archives(manifest_path: Path) -> list[Path]:
    payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    return [Path(path) for path in payload.get("archives") or []]


def iter_archives(dataset_dir: Path):
    for path in Path(dataset_dir).rglob("*"):
        if path.is_file() and any(path.name.lower().endswith(ext) for ext in ARCHIVE_EXTS):
            yield path


def resolve_dataset_archives(
    *, manifest_path: Path | None, dataset_dir: Path | None
) -> list[Path]:
    if manifest_path is not None:
        return load_manifest_archives(Path(manifest_path))
    if dataset_dir is not None:
        return sorted(iter_archives(Path(dataset_dir)))
    raise ValueError("Either manifest_path or dataset_dir must be provided.")


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _reset_dir(path: Path) -> Path:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _resolve_optional_path(value: str | None) -> Path | None:
    if not value:
        return None
    return Path(value).resolve()


def _unique_destination(dest_dir: Path, src: Path) -> Path:
    counter = 0
    while True:
        candidate_name = _bounded_destination_name(dest_dir, src.name, counter=counter)
        candidate = dest_dir / candidate_name
        if not candidate.exists():
            return candidate
        counter += 1


def _archive_name_parts(filename: str) -> tuple[str, str]:
    lower_name = filename.lower()
    for ext in sorted(ARCHIVE_EXTS, key=len, reverse=True):
        if lower_name.endswith(ext):
            return filename[: -len(ext)], filename[-len(ext) :]
    path = Path(filename)
    return path.stem, path.suffix


def _bounded_destination_name(dest_dir: Path, filename: str, *, counter: int) -> str:
    stem, suffix = _archive_name_parts(filename)
    counter_suffix = "" if counter == 0 else f"-{counter}"
    candidate_name = f"{stem}{counter_suffix}{suffix}"
    candidate_path = dest_dir / candidate_name
    if len(str(candidate_path)) <= WINDOWS_PATH_LIMIT:
        return candidate_name

    digest = hashlib.sha1(filename.encode("utf-8")).hexdigest()[:12]
    hash_suffix = f"-{digest}{counter_suffix}"
    max_name_length = max(1, WINDOWS_PATH_LIMIT - len(str(dest_dir)) - 1)
    max_stem_length = max(1, max_name_length - len(hash_suffix) - len(suffix))
    return f"{stem[:max_stem_length]}{hash_suffix}{suffix}"


def _materialize_file(src: Path, dest: Path, mode: str) -> None:
    try:
        if mode == "hardlink":
            os.link(src, dest)
        elif mode == "symlink":
            dest.symlink_to(src)
        elif mode == "copy":
            shutil.copy2(src, dest)
        else:
            raise ValueError(f"Unsupported materialize mode: {mode}")
    except OSError:
        shutil.copy2(src, dest)


def materialize_archives(archives: list[Path], dest_dir: Path, mode: str) -> list[Path]:
    _ensure_dir(dest_dir)
    materialized = []
    for archive in archives:
        destination = _unique_destination(dest_dir, archive)
        _materialize_file(archive, destination, mode)
        materialized.append(destination)
    return materialized


def sha1_short(path: Path, length: int = 8) -> str:
    sha1 = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            sha1.update(chunk)
    return sha1.hexdigest()[:length]


def text_sha1_short(value: str, length: int = 8) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:length]


def _chmod_and_retry(func, path, exc_info) -> None:
    try:
        os.chmod(path, stat.S_IWRITE | stat.S_IREAD | stat.S_IEXEC)
    except OSError:
        pass
    func(path)


def _remove_dir_if_exists(path: Path, *, ignore_errors: bool = False) -> None:
    if not path.exists():
        return
    last_exc = None
    for attempt in range(4):
        try:
            shutil.rmtree(path, onerror=_chmod_and_retry)
            return
        except Exception as exc:
            last_exc = exc
            if attempt < 3:
                time.sleep(0.25 * (attempt + 1))
    if not ignore_errors and last_exc is not None:
        raise last_exc


def _replace_dir_with_retry(src: Path, dest: Path) -> None:
    last_exc = None
    for attempt in range(6):
        try:
            _remove_dir_if_exists(dest)
            src.replace(dest)
            return
        except OSError as exc:
            last_exc = exc
            if attempt < 5 and getattr(exc, "winerror", None) == 5:
                time.sleep(0.25 * (attempt + 1))
                continue
            raise
    if last_exc is not None:
        raise last_exc


def safe_extract_tar(tar_path: Path, dest: Path):
    with tarfile.open(tar_path, "r:*") as tar:
        tar.extractall(dest)


def safe_extract_zip(zip_path: Path, dest: Path, pwd: str = "infected"):
    archive = zipfile.ZipFile(zip_path)
    try:
        archive.extractall(path=dest, pwd=pwd.encode())
    except RuntimeError:
        try:
            archive.extractall(path=dest)
        except Exception:
            subprocess.run(
                ["unzip", "-o", str(zip_path), "-d", str(dest)],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
    finally:
        archive.close()


def extract_archive_raw(archive_path: Path, tmpdir: Path) -> Path:
    outdir = tmpdir / f"p{sha1_short(archive_path, length=12)}"
    staging_dir = tmpdir / f"s{sha1_short(archive_path, length=10)}"
    _remove_dir_if_exists(outdir)
    _remove_dir_if_exists(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=False)
    try:
        if archive_path.suffix.lower() == ".zip" or archive_path.name.lower().endswith(".zip"):
            safe_extract_zip(archive_path, staging_dir)
        else:
            safe_extract_tar(archive_path, staging_dir)
        _replace_dir_with_retry(staging_dir, outdir)
    except Exception:
        _remove_dir_if_exists(staging_dir, ignore_errors=True)
        _remove_dir_if_exists(outdir, ignore_errors=True)
        raise
    return outdir


def decompress_packages(dataset_path: Path, output_root: Path) -> Path:
    output_root = _ensure_dir(output_root)
    temp_dataset_path = output_root / f"d{text_sha1_short(str(dataset_path), length=6)}"
    if temp_dataset_path.exists():
        shutil.rmtree(temp_dataset_path)
    temp_dataset_path.mkdir(parents=True, exist_ok=True)

    archives = list(iter_archives(dataset_path))
    failures = []
    for archive_path in archives:
        try:
            extract_archive_raw(archive_path, temp_dataset_path)
        except Exception as exc:
            failures.append((archive_path, exc))
    if failures:
        failure_records = [
            {"archive": str(archive_path), "reason": str(exc)}
            for archive_path, exc in failures
        ]
        (temp_dataset_path / "_extract_failures.json").write_text(
            json.dumps(failure_records, indent=2),
            encoding="utf-8",
        )
        failure_summary = "; ".join(
            f"{archive_path.name}: {exc}" for archive_path, exc in failures[:10]
        )
        print(
            f"[warn] Skipped {len(failures)} archive(s) during extraction: {failure_summary}",
            flush=True,
        )
    return temp_dataset_path


def _resolve_npm_command() -> str:
    return shutil.which("npm.cmd") or shutil.which("npm") or "npm"


def extract_features(dataset_dir: Path, feature_dir: Path, position_dir: Path, work_dir: Path) -> None:
    decompressed_dir = decompress_packages(dataset_dir, work_dir / "x")
    feature_dir.mkdir(parents=True, exist_ok=True)
    position_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            _resolve_npm_command(),
            "run",
            "start",
            "--",
            "-d",
            str(decompressed_dir),
            str(feature_dir),
            str(position_dir),
        ],
        cwd=str(Path(ROOT_PATH) / "feature-extract"),
        check=True,
    )


def _model_enum(model_name: str) -> ModelEnum:
    return {
        "RF": ModelEnum.RF,
        "MLP": ModelEnum.MLP,
        "NB": ModelEnum.NB,
        "SVM": ModelEnum.SVM,
    }[model_name]


def _preprocess_enum(name: str) -> PreprocessMethodEnum:
    return {
        "none": PreprocessMethodEnum.NONE,
        "standardlize": PreprocessMethodEnum.STANDARDLIZE,
        "min-max-scale": PreprocessMethodEnum.MIN_MAX_SCALE,
    }[name]


def default_hyperparameters(model_name: str) -> dict:
    if model_name == "RF":
        return {"number_of_decision_trees": 32, "maxium_depth": 11}
    if model_name == "NB":
        return {"smoothing": 1e-4}
    if model_name == "MLP":
        return {
            "learning_rate": 0.05045994670005887,
            "number_of_hidden_units": 16,
            "number_of_iterations": 400,
            "optimization": "adam",
            "activation": "logistic",
        }
    return {"gamma": "scale", "C": 1.070439127122467}


def _predict_one(feature_file: Path, model_name: str) -> str:
    if model_name == "RF":
        return predict_package_RF(str(feature_file))
    if model_name == "MLP":
        return predict_package_MLP(str(feature_file))
    if model_name == "NB":
        return predict_package_NB(str(feature_file))
    return predict_package_SVM(str(feature_file))


def evaluate_feature_dir(
    feature_dir: Path, model_name: str, true_label: str
) -> list[dict]:
    rows = []
    for feature_file in sorted(feature_dir.rglob("*.csv")):
        rows.append(
            {
                "sample": feature_file.stem,
                "prediction": _predict_one(feature_file, model_name),
                "true_label": true_label,
            }
        )
    return rows


def _metrics_from_predictions(rows: list[dict]) -> dict:
    y_true = [row["true_label"] for row in rows]
    y_pred = [row["prediction"] for row in rows]
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=["benign", "malicious"]
        ).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=["benign", "malicious"],
            target_names=["benign", "malicious"],
            digits=4,
            zero_division=0,
            output_dict=True,
        ),
        "sample_count": len(rows),
    }


def _copy_if_exists(src: str, dest: Path) -> str:
    source = Path(src)
    if source.exists():
        shutil.copy2(source, dest)
        return str(dest)
    return ""


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def add_standard_eval_parser(subparsers) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "standard-eval",
        help="run standard explicit split evaluation",
        description="Run MalPacDetector on explicit train/test manifests from research group splits.",
    )
    parser.add_argument("--split-dir", required=True)
    parser.add_argument("--benign-train-dir")
    parser.add_argument("--benign-train-manifest")
    parser.add_argument("--benign-test-dir")
    parser.add_argument("--benign-test-manifest")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--groundtruth-jsonl")
    parser.add_argument("--model", default="RF", choices=["NB", "MLP", "RF", "SVM"])
    parser.add_argument(
        "--preprocess",
        default="none",
        choices=["none", "standardlize", "min-max-scale"],
    )
    parser.add_argument(
        "--materialize",
        default="hardlink",
        choices=["copy", "hardlink", "symlink"],
    )
    parser.add_argument("--smote", action="store_true")
    return parser


def run_standard_eval(args: argparse.Namespace) -> dict:
    if not (args.benign_train_dir or args.benign_train_manifest):
        raise SystemExit("Provide either --benign-train-dir or --benign-train-manifest.")
    if not (args.benign_test_dir or args.benign_test_manifest):
        raise SystemExit("Provide either --benign-test-dir or --benign-test-manifest.")
    if args.model in {"NB", "MLP", "SVM"} and args.preprocess == "none":
        raise SystemExit("Use a scaler-backed preprocess for NB/MLP/SVM standard-eval runs.")

    split_dir = Path(args.split_dir).resolve()
    out_dir = _ensure_dir(Path(args.out_dir).resolve())
    work_dir = _reset_dir(out_dir / "w")
    datasets_dir = _reset_dir(work_dir / "d")
    features_root = _reset_dir(work_dir / "f")
    positions_root = _reset_dir(work_dir / "p")

    train_malicious = resolve_dataset_archives(
        manifest_path=split_dir / "train_manifest.json",
        dataset_dir=None,
    )
    test_malicious = resolve_dataset_archives(
        manifest_path=split_dir / "test_manifest.json",
        dataset_dir=None,
    )
    train_benign = resolve_dataset_archives(
        manifest_path=_resolve_optional_path(args.benign_train_manifest),
        dataset_dir=_resolve_optional_path(args.benign_train_dir),
    )
    test_benign = resolve_dataset_archives(
        manifest_path=_resolve_optional_path(args.benign_test_manifest),
        dataset_dir=_resolve_optional_path(args.benign_test_dir),
    )

    train_mal_dir = datasets_dir / "tm"
    test_mal_dir = datasets_dir / "em"
    train_ben_dir = datasets_dir / "tb"
    test_ben_dir = datasets_dir / "eb"
    materialize_archives(train_malicious, train_mal_dir, args.materialize)
    materialize_archives(test_malicious, test_mal_dir, args.materialize)
    materialize_archives(train_benign, train_ben_dir, args.materialize)
    materialize_archives(test_benign, test_ben_dir, args.materialize)

    train_mal_features = features_root / "tm"
    test_mal_features = features_root / "em"
    train_ben_features = features_root / "tb"
    test_ben_features = features_root / "eb"
    extract_features(train_mal_dir, train_mal_features, positions_root / "tm", work_dir)
    extract_features(test_mal_dir, test_mal_features, positions_root / "em", work_dir)
    extract_features(train_ben_dir, train_ben_features, positions_root / "tb", work_dir)
    extract_features(test_ben_dir, test_ben_features, positions_root / "eb", work_dir)

    train(
        [str(train_mal_features)],
        [str(train_ben_features)],
        _preprocess_enum(args.preprocess),
        _model_enum(args.model),
        ActionEnum.SAVE,
        default_hyperparameters(args.model),
        groundtruth_path=args.groundtruth_jsonl,
        smote=args.smote,
    )

    prediction_rows = []
    prediction_rows.extend(evaluate_feature_dir(test_mal_features, args.model, "malicious"))
    prediction_rows.extend(evaluate_feature_dir(test_ben_features, args.model, "benign"))

    predictions_csv = out_dir / "predictions.csv"
    with predictions_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample", "true_label", "prediction"])
        writer.writeheader()
        writer.writerows(prediction_rows)

    model_artifacts = {
        "RF": [rf_classifier_path, rf_scaler_save_path],
        "MLP": [MLP_path, mlp_scaler_save_path],
        "NB": [nb_path, nb_scaler_save_path],
        "SVM": [svm_path, svm_scaler_save_path],
    }[args.model]
    copied_models = []
    models_out = _ensure_dir(out_dir / "models")
    for source_path in model_artifacts:
        copied = _copy_if_exists(source_path, models_out / Path(source_path).name)
        if copied:
            copied_models.append(copied)

    metrics = _metrics_from_predictions(prediction_rows)
    payload = {
        "baseline": "malpacdetector",
        "split_dir": str(split_dir),
        "model": args.model,
        "preprocess": args.preprocess,
        "materialize_mode": args.materialize,
        "counts": {
            "train_malicious": len(train_malicious),
            "test_malicious": len(test_malicious),
            "train_benign": len(train_benign),
            "test_benign": len(test_benign),
        },
        "artifacts": {
            "predictions_csv": str(predictions_csv),
            "feature_root": str(features_root),
            "model_files": copied_models,
        },
        "metrics": metrics,
    }
    _write_json(out_dir / "metrics.json", payload)
    return payload
