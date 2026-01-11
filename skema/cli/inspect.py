# skema/cli/inspect.py

from pathlib import Path
import json
import typer

from ..core.dataset import Dataset
from ..core.schema import Schema, infer_schema

app = typer.Typer(help="Inspect dataset and infer schema")


# -------------------------
# Serialization helper
# -------------------------
def _schema_to_dict(schema: Schema) -> dict:
    """
    Serialize Schema dataclass into JSON-safe dict.
    Keeps core layer pure (no to_dict in schema).
    """
    return {
        "version": schema.version,
        "n_rows": schema.n_rows,
        "inferred_at": schema.inferred_at,
        "features": [
            {
                "name": f.name,
                "kind": f.kind,
                "role": f.role,
                "stats": {
                    "missing_pct": f.stats.missing_pct,
                    "unique_count": f.stats.unique_count,
                    "sample_values": f.stats.sample_values,
                },
                "flags": sorted(f.flags),
            }
            for f in schema.features
        ],
    }


# -------------------------
# Inspect command
# -------------------------
@app.command("inspect")
def inspect_cmd(
    data: Path = typer.Argument(..., help="Path to CSV file"),
    target: str | None = typer.Option(None, "--target", "-t", help="Target column name"),
    sample: int | None = typer.Option(None, "--sample", help="Sample first N rows"),
    outdir: Path = typer.Option(Path("./mlprep_artifacts"), "--outdir"),
):
    """
    Inspect a dataset and infer schema.
    """

    if not data.exists():
        typer.echo(f"❌ File not found: {data}")
        raise typer.Exit(code=1)

    dataset = Dataset.from_csv(data, sample=sample)
    df = dataset.df

    schema = infer_schema(df, target=target)

    # -------------------------
    # Console summary
    # -------------------------
    typer.echo(
        f"\n📊 Dataset Summary (Rows: {schema.n_rows}, Columns: {len(schema.features)})\n"
    )

    kind_counts: dict[str, int] = {}
    for f in schema.features:
        kind_counts[f.kind] = kind_counts.get(f.kind, 0) + 1

    for kind, count in kind_counts.items():
        typer.echo(f"{kind.capitalize():<12}: {count}")

    # -------------------------
    # Missing values
    # -------------------------
    typer.echo("\n🧩 Missing Values (>0%)")
    any_missing = False
    for f in schema.features:
        if f.stats.missing_pct > 0:
            any_missing = True
            typer.echo(f"- {f.name}: {f.stats.missing_pct * 100:.2f}%")

    if not any_missing:
        typer.echo("None")

    # -------------------------
    # Warnings (ACTIONABLE ONLY)
    # -------------------------
    ACTIONABLE_FLAGS = {
        "high_cardinality",
        "constant",
        "low_variance",
    }

    typer.echo("\n⚠️  Warnings")
    warned = False

    for f in schema.features:
        for flag in sorted(f.flags):
            if flag in ACTIONABLE_FLAGS:
                typer.echo(f"- {f.name}: {flag}")
                warned = True

    if not warned:
        typer.echo("None")

    # -------------------------
    # Artifacts
    # -------------------------
    outdir.mkdir(parents=True, exist_ok=True)

    with open(outdir / "schema.json", "w", encoding="utf-8") as f:
        json.dump(_schema_to_dict(schema), f, indent=2)

    # Placeholder for future stages
    with open(outdir / "warnings.json", "w", encoding="utf-8") as f:
        json.dump({}, f)

    typer.echo("\n✅ Inspection complete.\n")
