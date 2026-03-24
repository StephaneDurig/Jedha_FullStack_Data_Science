import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Union

from dotenv import load_dotenv

import mlflow
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

# Charge D99-PROJET/.env si présent (lancement depuis api/ ou racine du projet)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


def _sanitize_aws_credential_env(name: str) -> None:
    """
    Corrige des secrets mal collés (ex. valeur = « AWS_ACCESS_KEY_ID=AKIA... » ou retours à la ligne),
    ce qui provoque ValueError: Invalid header value côté botocore.
    """
    val = os.environ.get(name)
    if not val:
        return
    s = val.strip().replace("\r", "").replace("\n", "")
    prefix = f"{name}="
    if s.startswith(prefix):
        s = s[len(prefix) :].strip()
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1].strip()
    os.environ[name] = s


for _aws_key in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"):
    _sanitize_aws_credential_env(_aws_key)


def _local_mlflow_dir_uri(path: Path) -> str:
    """
    MLflow attend un URI file:// pour un store local. Un chemin Windows brut
    (ex. C:\\...\\mlruns) peut faire échouer silencieusement pyfunc.load_model (None).
    """
    return path.resolve().as_uri()


def _resolve_mlflow_tracking_uri() -> str:
    """
    Le notebook s'exécute depuis notebooks/ et écrit dans notebooks/mlruns.
    L'API est souvent lancée depuis api/ : un URI relatif \"mlruns\" pointerait vers api/mlruns
    (vide). On aligne par défaut sur notebooks/mlruns à la racine du projet.
    """
    raw = (os.environ.get("MLFLOW_TRACKING_URI") or "").strip()
    if raw.startswith("http://") or raw.startswith("https://"):
        return raw
    if raw.startswith("file:"):
        return raw
    if raw:
        p = Path(raw)
        if p.is_absolute() and p.exists():
            rp = p.resolve()
            return _local_mlflow_dir_uri(rp) if rp.is_dir() else str(rp)
    if not raw or raw in ("mlruns", "./mlruns", ".\\mlruns"):
        nb = _PROJECT_ROOT / "notebooks" / "mlruns"
        if nb.is_dir():
            return _local_mlflow_dir_uri(nb)
        root_mlruns = _PROJECT_ROOT / "mlruns"
        if root_mlruns.is_dir():
            return _local_mlflow_dir_uri(root_mlruns)
    if raw:
        candidate = _PROJECT_ROOT / raw
        if candidate.is_dir():
            return _local_mlflow_dir_uri(candidate)
    return raw or "mlruns"


# ---------------------------------------------------------------------------
# Configuration MLflow
# ---------------------------------------------------------------------------
MLFLOW_TRACKING_URI = _resolve_mlflow_tracking_uri()
MLFLOW_RUN_ID = os.environ.get("MLFLOW_RUN_ID", "")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Ordre identique au notebook notebooks/01-EDA_and_training.ipynb (ALL_FEATURES)
ALL_FEATURES: List[str] = [
    "model_key",
    "fuel",
    "paint_color",
    "car_type",
    "mileage",
    "engine_power",
    "private_parking_available",
    "has_gps",
    "has_air_conditioning",
    "automatic_car",
    "has_getaround_connect",
    "has_speed_regulator",
    "winter_tires",
]

NUM_COLS = len(ALL_FEATURES)


def _coerce_bool_feature(val: Any) -> int:
    if isinstance(val, bool):
        return int(val)
    if isinstance(val, (int, float)):
        return int(val)
    s = str(val).lower().strip()
    return 1 if s in ("1", "true", "yes") else 0


def _parse_row(row: List[Any]) -> dict:
    if len(row) != NUM_COLS:
        raise ValueError(
            f"Chaque ligne doit contenir {NUM_COLS} valeurs "
            f"(ordre : {', '.join(ALL_FEATURES)})."
        )
    out = {}
    for i, name in enumerate(ALL_FEATURES):
        val = row[i]
        if i < 4:
            out[name] = str(val)
        elif i < 6:
            # MLflow impose souvent int64 pour mileage / engine_power (float64 → erreur de schéma).
            out[name] = int(float(val))
        else:
            out[name] = _coerce_bool_feature(val)
    return out


# ---------------------------------------------------------------------------
# Description de l'API (affichée dans /docs)
# ---------------------------------------------------------------------------
description = """
# GetAround Pricing API

Cette API prédit le **prix journalier** (€/jour) à partir des caractéristiques véhicule,
via un pipeline sklearn + Gradient Boosting enregistré dans MLflow.

---

## `POST /predict`

Corps JSON attendu (clé **`input`** : une ou plusieurs lignes, chaque ligne est une liste
de **13 valeurs** dans l’ordre ci-dessous — même ordre que le notebook d’entraînement) :

1. `model_key` (texte)  
2. `fuel` (texte)  
3. `paint_color` (texte)  
4. `car_type` (texte)  
5. `mileage` (nombre)  
6. `engine_power` (nombre)  
7. à 13. booléens : `private_parking_available`, `has_gps`, `has_air_conditioning`,
   `automatic_car`, `has_getaround_connect`, `has_speed_regulator`, `winter_tires`
   (`true`/`false` ou `0`/`1`).

**Réponse :** `{"prediction": [<prix>, ...]}` (un prix par ligne, nombres flottants).

> L’énoncé du projet illustre aussi un exemple purement numérique (autre jeu de données).
> Ici, les types ci-dessus correspondent au fichier `get_around_pricing_project.csv`.

---

## `GET /docs`

Documentation interactive (Swagger UI).
"""

tags_metadata = [
    {"name": "General", "description": "Statut de l’API."},
    {"name": "Predictions", "description": "Prédictions de prix."},
]

# Référence partagée (conteneur mutable) : évite les soucis de portée async/global et de request.app.state.
_PYFUNC_MODEL: Dict[str, Any] = {"model": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Charge le modèle une fois au démarrage.
    On remplit un dict module (clé "model") + app.state pour redondance.
    """
    print(f"MLflow tracking URI (résolu) : {MLFLOW_TRACKING_URI}")
    rid = (MLFLOW_RUN_ID or "").strip()
    if not rid:
        raise RuntimeError(
            "MLFLOW_RUN_ID est vide. Renseignez-le dans .env (même ID que dans le notebook)."
        )
    logged_model = f"runs:/{rid}/getaround_pricing_model"
    loaded = mlflow.pyfunc.load_model(logged_model)
    if loaded is None:
        raise RuntimeError(
            "mlflow.pyfunc.load_model a renvoyé None. Pour un store local, utilisez un URI "
            "file:// (le projet résout automatiquement notebooks/mlruns en file://...)."
        )
    _PYFUNC_MODEL["model"] = loaded
    app.state.pricing_model = loaded
    print(f"Modèle chargé depuis : {logged_model}")
    yield
    _PYFUNC_MODEL["model"] = None
    if hasattr(app.state, "pricing_model"):
        try:
            delattr(app.state, "pricing_model")
        except Exception:
            pass


app = FastAPI(
    title="GetAround Pricing API",
    description=description,
    version="1.0.0",
    contact={
        "name": "GetAround Data Science",
        "url": "https://www.getaround.com",
    },
    openapi_tags=tags_metadata,
    lifespan=lifespan,
)


class PredictBody(BaseModel):
    """Corps attendu : une clé `input` contenant une liste de lignes."""

    input: List[List[Union[str, int, float, bool]]] = Field(
        ...,
        description="Liste de lignes ; chaque ligne = 13 valeurs dans l’ordre ALL_FEATURES.",
    )


@app.get("/", tags=["General"])
async def root():
    return {
        "message": "Welcome to the GetAround Pricing API",
        "status": "operational",
        "docs": "/docs",
        "predict": "/predict",
    }


@app.post("/predict", tags=["Predictions"])
async def predict(body: PredictBody, request: Request):
    """
    Prédit le prix journalier pour une ou plusieurs lignes (format `input` de l’énoncé).
    """
    model = _PYFUNC_MODEL.get("model") or getattr(
        request.app.state, "pricing_model", None
    )
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Modèle non chargé. Vérifiez les logs au démarrage.",
        )

    if not body.input:
        raise HTTPException(status_code=422, detail="Le champ 'input' ne doit pas être vide.")

    try:
        rows_dict = [_parse_row(list(row)) for row in body.input]
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    input_df = pd.DataFrame(rows_dict)
    preds = model.predict(input_df)
    prediction_list = [round(float(p), 2) for p in preds]
    return {"prediction": prediction_list}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
