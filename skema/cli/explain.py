# skema/cli/explain.py
from pathlib import Path
import typer
import json
from ..core.decisions import DecisionLog

def explain_cmd(
    outdir: Path = typer.Option(Path("./mlprep_artifacts"), "--outdir")
):
    """
    Show preprocessing decisions.
    """
    decisions = DecisionLog(outdir)
    decisions.load()
    if not decisions.decisions:
        typer.echo("No preprocessing decisions found. Run `mlprep run` first.")
        raise typer.Exit()

    typer.echo("\n📋 Preprocessing Decisions\n")
    for k, v in decisions.decisions.items():
        typer.echo(f"• {k}: {v}")
    typer.echo("\n✅ Done.\n")
