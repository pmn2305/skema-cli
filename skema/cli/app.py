# skema/cli/app.py
import typer
from .inspect import inspect_cmd
from .run import run_cmd
from .explain import explain_cmd
from .transform import transform_cmd

app = typer.Typer(help="skema — schema-first ML preprocessing")

app.command("inspect")(inspect_cmd)
app.command("run")(run_cmd)
app.command("explain")(explain_cmd)
app.command("transform")(transform_cmd)

if __name__ == "__main__":
    app()
