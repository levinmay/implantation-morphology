"""Four developmental groups and shared classifier construction."""

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import joblib
import sklearn
from .features import default_feature_set

CLASSES = (
    "early_pre_implantation_hatching",
    "floating_apposition_adhesion",
    "implantation_or_later",
    "degeneration",
)

SEED = 20260827


def four_group(parent: str, substage: str) -> str:
    if parent == "pre_implantation":
        if substage in {"early_pre_implantation", "hatching"}:
            return "early_pre_implantation_hatching"
        if substage == "floating_apposition_adhesion":
            return substage
        raise ValueError(f"Unexpected pre-implantation substage: {substage}")
    if parent in {"implantation_or_later", "degeneration"}:
        return parent
    raise ValueError(f"Unexpected parent state: {parent}")

def candidates(features: list[str], fold: int) -> list[tuple[str, Pipeline]]:
    models: list[tuple[str, Pipeline]] = []
    for c_value in (0.01, 0.1, 1.0, 10.0, 100.0):
        models.append((
            f"logistic_C{c_value:g}",
            Pipeline((
                ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scale", StandardScaler()),
                ("classify", LogisticRegression(
                    C=c_value,
                    class_weight="balanced",
                    max_iter=5000,
                    solver="lbfgs",
                    random_state=SEED + fold,
                )),
            )),
        ))
    for leaf in (2, 5, 10):
        for max_features in ("sqrt", 0.5):
            feature_label = "sqrt" if max_features == "sqrt" else "half"
            models.append((
                f"extra_trees_leaf{leaf}_{feature_label}",
                Pipeline((
                    ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                    ("classify", ExtraTreesClassifier(
                        n_estimators=500,
                        min_samples_leaf=leaf,
                        max_features=max_features,
                        class_weight="balanced",
                        random_state=SEED + fold,
                        n_jobs=4,
                    )),
                )),
            ))
    return models

def aligned_probability(estimator: Pipeline, values: pd.DataFrame) -> np.ndarray:
    probability = estimator.predict_proba(values)
    learned = estimator.named_steps["classify"].classes_.astype(int)
    aligned = np.zeros((len(values), len(CLASSES)), dtype=float)
    aligned[:, learned] = probability
    return aligned


def fit_model(frame: pd.DataFrame, model_name: str = "logistic_C1", seed_offset: int = 0) -> dict:
    """Fit a deployment model; this does not estimate held-out performance."""
    features = default_feature_set()["combined"]
    _check_features(frame, features)
    if "stage" not in frame or set(frame.stage) != set(CLASSES):
        raise ValueError("Training requires a stage column containing all four named groups")
    models = dict(candidates(features, seed_offset))
    if model_name not in models:
        raise ValueError(f"Unknown model; choose one of {sorted(models)}")
    estimator = models[model_name]
    estimator.fit(frame[features], frame.stage.map({name: i for i, name in enumerate(CLASSES)}))
    return {"schema_version": 1, "features": features, "classes": list(CLASSES),
            "sklearn_version": sklearn.__version__, "model_name": model_name,
            "seed": SEED + seed_offset, "estimator": estimator}


def _check_features(frame: pd.DataFrame, features: list[str]) -> None:
    if frame.empty or not frame.columns.is_unique or not set(features).issubset(frame):
        raise ValueError("Expected nonempty rows with unique, complete feature columns")
    values = frame[features].to_numpy(dtype=float)
    if np.isinf(values).any() or np.isnan(values).all(axis=1).any():
        raise ValueError("Infinite or entirely missing feature rows cannot be classified")


def save_model(model: dict, destination: Path) -> None:
    """Save estimator, feature order, class mapping, and library version together."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as handle:
        joblib.dump(model, handle)


def load_model(path: Path) -> dict:
    """Load only trusted files: joblib uses pickle and can execute code."""
    model = joblib.load(path)
    if (model.get("schema_version") != 1 or model.get("classes") != list(CLASSES)
            or model.get("features") != default_feature_set()["combined"]
            or model.get("sklearn_version") != sklearn.__version__):
        raise ValueError("Incompatible staging bundle: schema, classes, features, or sklearn version")
    return model


def predict_stages(model: dict, frame: pd.DataFrame) -> pd.DataFrame:
    features = model["features"]
    _check_features(frame, features)
    probability = aligned_probability(model["estimator"], frame[features])
    result = pd.DataFrame(probability, columns=[f"prob_{c}" for c in CLASSES], index=frame.index)
    result.insert(0, "stage", np.asarray(CLASSES)[probability.argmax(axis=1)])
    return result
