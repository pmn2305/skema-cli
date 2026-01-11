# skema/core/decisions.py
import json
from pathlib import Path

class DecisionLog:
    def __init__(self, outdir: Path):
        self.outdir = Path(outdir)
        self.decisions = {}
        self.outdir.mkdir(parents=True, exist_ok=True)

    def log(self, key, value):
        self.decisions[key] = value

    def save(self, filename="decisions.json"):
        with open(self.outdir / filename, "w") as f:
            json.dump(self.decisions, f, indent=2)

    def load(self, filename="decisions.json"):
        path = self.outdir / filename
        if path.exists():
            with open(path) as f:
                self.decisions = json.load(f)
        return self.decisions
