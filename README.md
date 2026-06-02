# MalPacDetector Reproduction

This repository is a runnable reproduction of:

> MalPacDetector: An LLM-based Malicious npm Package Detector

MalPacDetector combines Node.js feature extraction with Python machine-learning classifiers for NPM malicious-package detection. The standardized evaluation entry point is:

```text
cli.py standard-eval
```

It consumes explicit train/test split manifests, extracts features, trains the selected classifier, evaluates on the test split, and writes `metrics.json`.

## Environment

Recommended versions:

- Windows or Ubuntu 22.04
- Python 3.10
- Node.js 18.x or newer
- npm

### Windows PowerShell

```powershell
uv python install 3.10
uv sync

cd .\feature-extract
npm install
npm run compile
cd ..
```

### Linux / WSL

```bash
uv python install 3.10
uv sync

cd feature-extract
npm install
npm run compile
cd ..
```

`.venv/`, `env/`, feature outputs, reports, and results are ignored by git and can remain in the repository.

## Configuration

The original project uses `conf/settings.json` for default locations:

- `datasets`: package dataset root
- `models`: saved model directory
- `reports`: prediction reports
- `features`: extracted feature files
- `feature-positions`: source-code position records

For standardized evaluation, `standard-eval` creates and manages its own output directory, so you usually do not need to run `configure.py`.

## Inputs

Required inputs for standard evaluation:

- `--split-dir`: contains `train_manifest.json` and `test_manifest.json`.
- `--benign-train-dir` or `--benign-train-manifest`.
- `--benign-test-dir` or `--benign-test-manifest`.
- `--out-dir`: output directory.

Optional:

- `--groundtruth-jsonl`: annotation JSONL used for subgroup reports.
- `--model`: `NB`, `MLP`, `RF`, or `SVM`.
- `--preprocess`: `none`, `standardlize`, or `min-max-scale`.

## Run Standard Evaluation

PowerShell:

```powershell
uv run .\cli.py standard-eval `
  --split-dir C:\path\to\split `
  --benign-train-dir C:\path\to\benign\train `
  --benign-test-dir C:\path\to\benign\test `
  --groundtruth-jsonl C:\path\to\annotations.jsonl `
  --out-dir .\results\standard_eval `
  --model RF `
  --preprocess none `
  --materialize hardlink
```

Linux / WSL:

```bash
uv run cli.py standard-eval \
  --split-dir /path/to/split \
  --benign-train-dir /path/to/benign/train \
  --benign-test-dir /path/to/benign/test \
  --groundtruth-jsonl /path/to/annotations.jsonl \
  --out-dir ./results/standard_eval \
  --model RF \
  --preprocess none \
  --materialize hardlink
```

Useful options:

- `--materialize`: `copy`, `hardlink`, or `symlink`.
- `--smote`: enable SMOTE oversampling.
- `--model`: `RF` is the default; other supported values are `NB`, `MLP`, and `SVM`.

## Outputs

The output directory contains materialized package sets, feature files, model artifacts, predictions, and:

```text
metrics.json
```

`metrics.json` contains the binary classification metrics for the selected split.

## Original CLI

The original project CLI is still available:

```powershell
uv run .\cli.py -h
uv run .\cli.py extract -h
uv run .\cli.py train -h
uv run .\cli.py predict -h
```

Typical original workflow:

```powershell
uv run .\cli.py extract -d <dataset_name>
uv run .\cli.py train -a training -m <malicious_dataset_name> -b <benign_dataset_name> -p none -o RF
uv run .\cli.py predict -o RF -d <dataset_name>
```

For controlled comparisons, prefer `standard-eval`.

## Common Issues

- `npm run compile` fails: make sure Node.js and npm are installed and rerun inside `feature-extract/`.
- Feature extraction produces no rows: confirm archives are valid npm package archives and contain `package/package.json` or a recognizable package root.
- Hardlink materialization fails across drives: use `--materialize copy`.
- Existing `models/*.pkl` are tracked legacy artifacts; `.gitignore` only affects new untracked model outputs.
