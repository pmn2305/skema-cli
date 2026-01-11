# skema/cli/run.py
from pathlib import Path
import typer
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import joblib

from ..core.dataset import Dataset
from ..core.schema import infer_schema
from ..core.decisions import DecisionLog

def run_cmd(
    data: Path = typer.Argument(..., help="Path to CSV file"),
    target: str = typer.Option(..., "--target", "-t", help="Target column"),
    task: str = typer.Option(..., "--task", help="classification|regression|timeseries"),
    model: str = typer.Option("linear", "--model", help="linear|tree|nn"),
    test_size: float = typer.Option(0.2, "--test-size"),
    val_size: float = typer.Option(0.1, "--val-size"),
    missing_numeric: str = typer.Option("median", "--missing-numeric"),
    missing_categorical: str = typer.Option("most_frequent", "--missing-categorical"),
    encode: str = typer.Option("onehot", "--encode"),
    scale: str = typer.Option("standard", "--scale"),
    outdir: Path = typer.Option(Path("./mlprep_artifacts"), "--outdir")
):
    if not data.exists():
        typer.echo(f"❌ File not found: {data}")
        raise typer.Exit(code=1)

    typer.echo(f"📂 Loading dataset from: {data}")
    dataset = Dataset.from_csv(data)
    df = dataset.df

    typer.echo("📊 Inferring schema...")
    schema = infer_schema(df, target=target)

    # Drop target to get feature columns
    X = df.drop(columns=[target])
    y = df[target]

    # Get columns by type but ensure they exist in X
    numeric_cols = [f.name for f in schema.features if f.kind=="numeric" and f.name in X.columns]
    cat_cols = [f.name for f in schema.features if f.kind=="categorical" and f.name in X.columns]

    if not numeric_cols and not cat_cols:
        typer.echo("❌ No valid feature columns found. Exiting.")
        raise typer.Exit(code=1)

    typer.echo(f"🔹 Numeric columns: {numeric_cols}")
    typer.echo(f"🔹 Categorical columns: {cat_cols}")

    transformers = []

    if numeric_cols:
        num_pipeline = Pipeline([
            ("impute", SimpleImputer(strategy=missing_numeric)),
            ("scale", StandardScaler() if scale=="standard" and model in ["linear","nn"] else "passthrough")
        ])
        transformers.append(("num", num_pipeline, numeric_cols))

    if cat_cols:
        cat_pipeline = Pipeline([
            ("impute", SimpleImputer(strategy=missing_categorical)),
            ("encode", OneHotEncoder(handle_unknown="ignore") if encode=="onehot" else "passthrough")
        ])
        transformers.append(("cat", cat_pipeline, cat_cols))

    preprocessor = ColumnTransformer(transformers)

    typer.echo(f"🔹 Splitting dataset for task: {task}")
    if task == "classification":
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size + val_size, random_state=42)
        for train_val_idx, test_idx in splitter.split(X, y):
            pass
        val_relative = val_size / (test_size + val_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X.iloc[train_val_idx], y.iloc[train_val_idx],
            test_size=val_relative,
            stratify=y.iloc[train_val_idx],
            random_state=42
        )
        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size/(1-test_size), random_state=42)

    typer.echo(f"📌 Dataset sizes -> Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    typer.echo("⚙️ Fitting preprocessing pipeline...")
    preprocessor.fit(X_train)


# Transform a few rows for sanity check
    X_train_transformed = preprocessor.transform(X_train)
    if hasattr(X_train_transformed, "toarray"):  # handle sparse matrix
       X_train_transformed = X_train_transformed.toarray()

# Extract feature names
    num_features = numeric_cols
    if encode == "onehot" and cat_cols:
       cat_features = preprocessor.named_transformers_['cat']['encode'].get_feature_names_out(cat_cols)
    else:
       cat_features = cat_cols
    all_features = list(num_features) + list(cat_features)

# Show first 5 transformed rows as a nice DataFrame
    typer.echo("\n🔍 Sample transformed data (first 5 rows):")
    typer.echo(pd.DataFrame(X_train_transformed, columns=all_features).head())


    outdir.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, outdir / "preprocessor.pkl")
    np.savez(outdir / "split_indices.npz",
             train_idx=X_train.index.values,
             val_idx=X_val.index.values,
             test_idx=X_test.index.values)

    decisions = DecisionLog(outdir)
    decisions.load()
    decisions.log("preprocessor", {
        "numeric_imputer": missing_numeric,
        "categorical_imputer": missing_categorical,
        "encoding": encode,
        "scaling": scale,
        "model_type": model,
        "task": task
    })
    decisions.save()

    typer.echo("\n✅ Preprocessing run complete.")
    typer.echo(f"Artifacts saved to: {outdir}")
    typer.echo("✔️ Pipeline ready for modeling.\n")
