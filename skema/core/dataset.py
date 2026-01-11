# skema/core/dataset.py
import pandas as pd
from pathlib import Path

class Dataset:
    def __init__(self, df: pd.DataFrame, path: Path | None = None):
        self.df = df
        self.path = path

    @classmethod
    def from_csv(cls, path: str | Path, sample: int | None = None):
        path = Path(path)
        df = pd.read_csv(path)
        if sample:
            df = df.head(sample)
        return cls(df, path)

    def summary(self):
        rows, cols = self.df.shape
        types = self.df.dtypes.to_dict()
        missing = self.df.isna().mean().to_dict()
        return {"rows": rows, "columns": cols, "types": types, "missing": missing}
