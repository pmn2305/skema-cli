
# **skema-cli** 🚀

[![Python Version](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**skema-cli** is a **command-line tool for schema-first machine learning preprocessing**. It allows you to inspect datasets, automatically generate preprocessing pipelines, explain features, and transform datasets—all from the CLI. Perfect for data scientists, ML engineers, and AIML students who want **quick, reproducible preprocessing** workflows.

---

## **Features**

* Inspect your dataset: summary stats, missing values, and schema suggestions
* Run preprocessing pipelines with automatic handling of numeric & categorical features
* Explain feature importance and contribution to target variables
* Transform datasets using a saved preprocessor
* Fully CLI-driven, no heavy IDE required

---

## **Installation**

```bash
git clone https://github.com/pmn2305/skema-cli.git
cd skema-cli
pip install -r requirements.txt
```

> Requires Python 3.10+

---

## **Commands & Usage**

All commands are available via:

```bash
python -m skema.cli.app <command> [options]
```

### 1️⃣ Inspect your dataset

```bash
python -m skema.cli.app inspect <dataset.csv>
```

* Displays dataset summary: column types, missing values, and schema suggestions.
* Helps you understand your data before preprocessing.

---

### 2️⃣ Run preprocessing pipeline

```bash
python -m skema.cli.app run <dataset.csv> --target <target_column> --task <classification|regression|timeseries>
```

Optional flags:

* `--model` : `linear` | `tree` | `nn` (default: `linear`)
* `--test-size` : fraction of data for test split (default: 0.2)
* `--val-size` : fraction of training data for validation (default: 0.1)
* `--missing-numeric` : `median` | `mean` | `constant` (default: `median`)
* `--missing-categorical` : `most_frequent` | `constant` (default: `most_frequent`)
* `--encode` : `onehot` | `passthrough` (default: `onehot`)
* `--scale` : `standard` | `passthrough` (default: `standard`)
* `--outdir` : directory to save preprocessor and split indices (default: `./mlprep_artifacts`)

Outputs:

* `preprocessor.pkl` → saved preprocessing pipeline
* `split_indices.npz` → train/val/test indices
* Logs preprocessing decisions automatically

---

### 3️⃣ Explain features

```bash
python -m skema.cli.app explain <dataset.csv> --target <target_column>
```

* Generates feature importance and contribution explanations.
* Ideal for quickly understanding which features matter most.

---

### 4️⃣ Transform new data

```bash
python -m skema.cli.app transform <dataset.csv> --preprocessor-path <path_to_preprocessor.pkl> --output-path <output.csv>
```

Optional flags:

* `--no-save-splits` : skips saving the transformed CSV
* Transforms datasets using a previously saved preprocessor.

---

## **Example Workflow**

```bash
# Step 1: Inspect dataset
python -m skema.cli.app inspect datademo.csv

# Step 2: Run preprocessing
python -m skema.cli.app run datademo.csv --target churn --task classification

# Step 3: Transform new dataset
python -m skema.cli.app transform datademo.csv --preprocessor-path mlprep_artifacts/preprocessor.pkl --output-path datademo_transformed.csv
```

---

## **Directory Structure**

```
skema-cli/
├─ skema/
│  ├─ cli/
│  │  ├─ app.py
│  │  ├─ inspect.py
│  │  ├─ run.py
│  │  ├─ explain.py
│  │  └─ transform.py
│  ├─ core/
│  │  ├─ dataset.py
│  │  ├─ schema.py
│  │  └─ decisions.py
├─ mlprep_artifacts/   # generated after `run` command
├─ datademo.csv        # sample dataset
├─ README.md
└─ requirements.txt
```

---

## **License**

MIT License © 2026 Prerana M N


