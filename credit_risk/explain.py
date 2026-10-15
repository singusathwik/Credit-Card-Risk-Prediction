"""Model-agnostic SHAP of the EXACT served class-1 probability, calibration included."""

import numpy as np
import pandas as pd
import shap
from .features import RAW


def explain(model, background, X, permutations=3):
    def probability(values):
        return model.predict_proba(pd.DataFrame(values, columns=RAW))[:, 1]

    explainer = shap.PermutationExplainer(probability, background[RAW], seed=42)
    return explainer(X[RAW], max_evals=(2 * len(RAW) + 1) * permutations, silent=True)


def local_explanation(bundle, X):
    e = explain(bundle["model"], bundle["background"], X)
    values = e.values[0]
    return {
        "model_version": bundle["metadata"]["model_version"],
        "output": "class_1_probability",
        "units": "probability",
        "base_value": float(e.base_values[0]),
        "reconstructed_probability": float(e.base_values[0] + values.sum()),
        "method": "permutation SHAP, 3 permutations, training background",
        "contributions": sorted(
            [
                {"feature": n, "value": float(X.iloc[0][n]), "contribution": float(v)}
                for n, v in zip(RAW, values)
            ],
            key=lambda r: abs(r["contribution"]),
            reverse=True,
        ),
        "limitation": "Associations, not causes. Correlated features and background choice affect attributions.",
    }
