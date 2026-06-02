#!/usr/bin/env python3
import argparse
import csv
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

from conf import SETTINGS
from training.src.model_util import (
    build_classification_report,
    compute_type_detection_rates,
    evaluate_model,
)
from training.src.groundtruth import load_malicious_type_map, _normalize_package_name


POSITIVE_VALUES = {"1", "true", "yes", "y", "malicious", "pos", "positive"}


def normalize_package_name(value: str) -> str:
    return _normalize_package_name(value)


def load_malicious_set(path: Path, package_col: str, label_col: str | None):
    if path.suffix.lower() == ".jsonl":
        malicious = set()
        with path.open("r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue
                record = json.loads(line)
                verdict = (record.get("annotation") or {}).get("verdict")
                if verdict and str(verdict).lower() != "malicious":
                    continue
                archive_name = record.get("archive_name") or record.get("src_archive")
                if archive_name:
                    malicious.add(normalize_package_name(archive_name))
        if malicious:
            return malicious
        type_map, _ = load_malicious_type_map(str(path))
        return {normalize_package_name(k) for k in type_map.keys()}

    malicious = set()
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"Missing header in {path}")
        header = reader.fieldnames
        if package_col not in header:
            if "package" in header:
                package_col = "package"
            elif "name" in header:
                package_col = "name"
            else:
                raise ValueError(f"Missing package column in {path}")
        if label_col is None:
            if "label" in header:
                label_col = "label"
            elif "is_malicious" in header:
                label_col = "is_malicious"
            elif "malicious" in header:
                label_col = "malicious"

        for row in reader:
            pkg = normalize_package_name(row.get(package_col, ""))
            if not pkg:
                continue
            if label_col is None:
                malicious.add(pkg)
            else:
                value = str(row.get(label_col, "")).strip().lower()
                if value in POSITIVE_VALUES:
                    malicious.add(pkg)
    return malicious


def list_feature_files(dir_path: Path):
    return [p for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() == ".csv"]


def split_packages(files, test_ratio: float, seed: int):
    packages = [p.stem for p in files]
    rng = random.Random(seed)
    rng.shuffle(packages)
    test_count = max(1, int(len(packages) * test_ratio)) if packages else 0
    test_pkgs = set(packages[:test_count])
    train_files = [p for p in files if p.stem not in test_pkgs]
    test_files = [p for p in files if p.stem in test_pkgs]
    return train_files, test_files, len(test_pkgs), len(packages)


def copy_files(files, dest_dir: Path):
    dest_dir.mkdir(parents=True, exist_ok=True)
    for src in files:
        shutil.copy2(src, dest_dir / src.name)


def ensure_dataset_dir(name: str, base_dir: Path):
    dest = base_dir / name
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)
    return dest


def run_cli(args_list):
    cmd = [sys.executable, "cli.py"] + args_list
    subprocess.run(cmd, check=True)


def read_report(report_path: Path):
    rows = []
    with report_path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            return rows
        for row in reader:
            if not row:
                continue
            name = row[0].strip()
            pred = row[1].strip() if len(row) > 1 else ""
            rows.append((name, pred))
    return rows


def main():
    parser = argparse.ArgumentParser(description="Split features, train via cli.py, predict, and report metrics.")
    parser.add_argument("--inputs", nargs="+", required=True, help="Malicious feature dataset names or paths.")
    parser.add_argument("--negatives", required=True, help="Benign feature dataset name or path.")
    parser.add_argument("--groundtruth", help="Ground truth file (csv/jsonl) for malicious packages.")
    parser.add_argument("--model", default="NB", help="Model name (NB/MLP/RF/SVM).")
    parser.add_argument("--preprocess", default="none", help="Preprocess method.")
    parser.add_argument("--action", default="training", help="Train action: training or save.")
    parser.add_argument("--save-model", action="store_true", help="Save model after training.")
    parser.add_argument("--split-ratio", type=float, default=0.2, help="Test ratio by package.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--prefix", default="difficulty", help="Prefix for generated dataset dirs.")
    parser.add_argument("--gt-package-col", default="package_name", help="Package column in groundtruth CSV.")
    parser.add_argument("--gt-label-col", default=None, help="Optional label column in groundtruth CSV.")
    parser.add_argument("--results-dir", default="results", help="Directory for metrics output.")
    parser.add_argument("--smote", action="store_true", help="Apply SMOTE during training.")
    parser.add_argument(
        "--simple",
        action="store_true",
        help="Use full datasets for training/testing without difficulty split.",
    )
    args = parser.parse_args()

    features_root = Path(SETTINGS["path"]["features"])
    reports_root = Path(SETTINGS["path"]["reports"])
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    def resolve_dataset(value: str) -> Path:
        candidate = Path(value)
        if candidate.exists():
            return candidate
        return features_root / value

    inputs = [resolve_dataset(p) for p in args.inputs]
    negatives = resolve_dataset(args.negatives)
    malicious_set = set()
    type_map = {}
    if args.groundtruth:
        groundtruth_path = Path(args.groundtruth)
        malicious_set = load_malicious_set(groundtruth_path, args.gt_package_col, args.gt_label_col)
        type_map, _ = load_malicious_type_map(str(groundtruth_path))
    for src_dir in inputs:
        for feature_file in list_feature_files(src_dir):
            malicious_set.add(feature_file.stem)

    mal_train_name = f"{args.prefix}_mal_train"
    ben_train_name = f"{args.prefix}_ben_train"
    mal_train_dir = ensure_dataset_dir(mal_train_name, features_root)
    ben_train_dir = ensure_dataset_dir(ben_train_name, features_root)

    test_dataset_names = []
    if args.simple:
        print("==> Splitting datasets into train/test (simple mode)")
        neg_files = list_feature_files(negatives)
        neg_train, neg_test, neg_test_pkgs, neg_total_pkgs = split_packages(
            neg_files, args.split_ratio, args.seed
        )
        print(f"Negatives: test packages {neg_test_pkgs}/{neg_total_pkgs}")
        copy_files(neg_train, ben_train_dir)

        for src_dir in inputs:
            files = list_feature_files(src_dir)
            train_files, test_files, test_pkgs, total_pkgs = split_packages(
                files, args.split_ratio, args.seed
            )
            copy_files(train_files, mal_train_dir)

            dataset_name = f"{args.prefix}_{src_dir.name}_simple_test"
            test_dir = ensure_dataset_dir(dataset_name, features_root)
            copy_files(test_files, test_dir)
            copy_files(neg_test, test_dir)
            test_dataset_names.append(dataset_name)
            print(f"Wrote {test_dir} (test packages: {test_pkgs}/{total_pkgs})")
    else:
        print("==> Splitting positives and negatives into train/test (by package name)")
        neg_files = list_feature_files(negatives)
        neg_train, neg_test, neg_test_pkgs, neg_total_pkgs = split_packages(
            neg_files, args.split_ratio, args.seed
        )
        print(f"Negatives: test packages {neg_test_pkgs}/{neg_total_pkgs}")
        copy_files(neg_train, ben_train_dir)

        for src_dir in inputs:
            files = list_feature_files(src_dir)
            train_files, test_files, test_pkgs, total_pkgs = split_packages(
                files, args.split_ratio, args.seed
            )
            copy_files(train_files, mal_train_dir)

            dataset_name = f"{args.prefix}_{src_dir.name}_test"
            test_dir = ensure_dataset_dir(dataset_name, features_root)
            copy_files(test_files, test_dir)
            copy_files(neg_test, test_dir)
            test_dataset_names.append(dataset_name)
            print(f"Wrote {test_dir} (test packages: {test_pkgs}/{total_pkgs})")

    print("==> Training model via cli.py")
    train_args = [
        "train",
        "-a", args.action,
        "-m", mal_train_name,
        "-b", ben_train_name,
        "-p", args.preprocess,
        "-o", args.model,
    ]
    if args.groundtruth:
        train_args += ["-g", args.groundtruth]
    if args.smote:
        train_args.append("--smote")
    run_cli(train_args)

    if args.save_model and args.action != "save":
        print("==> Saving model via cli.py")
        model = args.model
        save_args = [
            "train",
            "-a", "save",
            "-m", mal_train_name,
            "-b", ben_train_name,
            "-p", args.preprocess,
            "-o", model,
        ]
        if args.groundtruth:
            save_args += ["-g", args.groundtruth]
        if args.smote:
            save_args.append("--smote")
        if model == "NB":
            smoothing = SETTINGS["classifier"]["hyperparameters"]["NB"]["smoothings"][0]
            save_args += ["-hs", str(smoothing)]
        elif model == "MLP":
            hp = SETTINGS["classifier"]["hyperparameters"]["MLP"]
            save_args += [
                "-hr", str(hp["learning_rates"][0]),
                "-hl", str(hp["number_of_hidden_units"][0]),
                "-hi", str(hp["number_of_iterations"][0]),
                "-ho", str(hp["optimization_algorithms"][0]),
                "-ha", str(hp["activation_functions"][0]),
            ]
        elif model == "RF":
            hp = SETTINGS["classifier"]["hyperparameters"]["RF"]
            save_args += [
                "-he", str(hp["number_of_decision_trees"][0]),
                "-hd", str(hp["maxium_depths"][0]),
            ]
        elif model == "SVM":
            hp = SETTINGS["classifier"]["hyperparameters"]["SVM"]
            save_args += [
                "-hg", str(hp["gammas"][0]),
                "-hc", str(hp["C"][0]),
            ]
        run_cli(save_args)

    type_totals = {}
    type_detected = {}

    for dataset_name in test_dataset_names:
        print(f"==> Predicting: {dataset_name}")
        run_cli(["predict", "-o", args.model, "-d", dataset_name])
        report_path = reports_root / f"{dataset_name}-{args.model}-report.csv"
        rows = read_report(report_path)
        if not rows:
            print(f"Warning: empty report {report_path}")
            continue

        y_true = []
        y_pred = []
        type_labels = []
        for pkg, pred in rows:
            y_pred.append(pred)
            norm_pkg = normalize_package_name(pkg)
            y_true.append("malicious" if norm_pkg in malicious_set else "benign")
            type_labels.append(type_map.get(norm_pkg, []))

        metrics = evaluate_model(y_true, y_pred)
        print(build_classification_report(y_true, y_pred))
        metric_header = ["tp", "fp", "tn", "fn", "acc", "prec", "rec", "f1", "mcc"]
        metrics_dict = {
            key: (value.item() if hasattr(value, "item") else value)
            for key, value in zip(metric_header, metrics)
        }
        metrics_out = results_dir / f"{dataset_name}-{args.model}-metrics.json"
        with metrics_out.open("w", encoding="utf-8") as f:
            json.dump(metrics_dict, f, indent=2)
        print(metrics_dict)

        type_rows = compute_type_detection_rates(y_true, y_pred, type_labels)
        for malicious_type, detected, total, _ in type_rows:
            type_totals[malicious_type] = type_totals.get(malicious_type, 0) + total
            type_detected[malicious_type] = type_detected.get(malicious_type, 0) + detected
        print()

    if type_totals:
        print("==> Type metrics (all test sets combined)")
        print("type, detected, total, rate")
        for malicious_type in sorted(type_totals.keys()):
            total = type_totals[malicious_type]
            detected = type_detected.get(malicious_type, 0)
            rate = detected / total if total else 0.0
            print(f"{malicious_type}, {detected}, {total}, {rate:.6f}")


if __name__ == "__main__":
    main()
