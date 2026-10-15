from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from .features import RAW

TARGET = "default payment next month"


def load_data(path):
    d = pd.read_excel(path, header=1)
    required = RAW + ["ID", TARGET]
    if not set(required).issubset(d.columns):
        raise ValueError("Dataset does not match the original UCI schema.")
    if d[TARGET].isna().any() or set(d[TARGET].unique()) != {0, 1}:
        raise ValueError("Target must contain binary labels 0 and 1.")
    return d


def groups_for(d):
    # Connected components: identical raw profiles OR same customer ID stay together.
    parent = np.arange(len(d))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for keys in [
        d.ID.tolist(),
        pd.util.hash_pandas_object(d[RAW], index=False).tolist(),
    ]:
        seen = {}
        for i, key in enumerate(keys):
            if key in seen:
                parent[root(i)] = root(seen[key])
            else:
                seen[key] = i
    return np.array([root(i) for i in range(len(d))])


def split_data(d, seed=42):
    groups = groups_for(d)
    y = d[TARGET].to_numpy()
    rest, test = next(
        StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(d, y, groups)
    )
    a, b = next(
        StratifiedGroupKFold(4, shuffle=True, random_state=seed + 1).split(
            d.iloc[rest], y[rest], groups[rest]
        )
    )
    return {"train": rest[a], "validation": rest[b], "test": test}, groups


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
