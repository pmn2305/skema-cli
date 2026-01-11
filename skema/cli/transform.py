# skema/cli/transform.py
import pandas as pd
import joblib
from pathlib import Path
import typer

def transform_cmd(
    input_file: Path,
    preprocessor_path: Path = Path("mlprep_artifacts/preprocessor.pkl"),
    no_save_splits: bool = False,
    output_path: Path = Path("transformed_data.csv")
):
    """Transform a dataset using a saved preprocessor."""
    # Load dataset
    df = pd.read_csv(input_file)
    typer.echo(f"📂 Loaded dataset from: {input_file}")

    # Load preprocessor
    preprocessor = joblib.load(preprocessor_path)
    typer.echo(f"⚙️ Loaded preprocessor from: {preprocessor_path}")

    # Separate features and target if target exists
    y_cols = ["churn"] if "churn" in df.columns else []
    X = df.drop(columns=y_cols) if y_cols else df.copy()

    # Transform
    try:
        X_transformed = preprocessor.transform(X)
        if hasattr(X_transformed, "toarray"):  # sparse output
            X_transformed = X_transformed.toarray()

        # Automatically get output feature names
        try:
            all_features = preprocessor.get_feature_names_out()
        except Exception:
            all_features = [f"f{i}" for i in range(X_transformed.shape[1])]

        df_transformed = pd.DataFrame(X_transformed, columns=all_features)

        # Attach target columns back if they exist
        if y_cols:
            df_transformed[y_cols] = df[y_cols].values

        typer.echo(f"✅ Transformation complete. Shape: {df_transformed.shape}")

        # Save transformed data
        if not no_save_splits:
            df_transformed.to_csv(output_path, index=False)
            typer.echo(f"💾 Saved transformed data to: {output_path}")
        else:
            typer.echo("⚠️ Skipping save due to --no-save-splits flag.")

    except Exception as e:
        typer.echo(f"❌ Error during transformation: {e}")
