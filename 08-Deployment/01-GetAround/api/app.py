import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Literal

from dotenv import load_dotenv

import mlflow
import pandas as pd
import uvicorn
from fastapi import Body, FastAPI, HTTPException, Request
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
MLFLOW_REGISTERED_MODEL = os.environ.get("MLFLOW_REGISTERED_MODEL", "").strip()
MLFLOW_MODEL_ALIAS = os.environ.get("MLFLOW_MODEL_ALIAS", "").strip()
MLFLOW_ARTIFACT_PATH = os.environ.get("MLFLOW_ARTIFACT_PATH", "getaround_pricing_model").strip()

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

# ---------------------------------------------------------------------------
# Description de l'API (affichée dans /docs)
# ---------------------------------------------------------------------------
description = (
    "Prédit le **prix journalier** (€/jour) d'un véhicule GetAround à partir de "
    "ses caractéristiques (marque, carburant, kilométrage, équipements…).\n\n"
    "Modèle : pipeline `scikit-learn` + Gradient Boosting entraîné sur "
    "`get_around_pricing_project.csv` et versionné dans MLflow "
    "(`ZerphirosX/mlflow`)."
)

tags_metadata = [
    {"name": "General", "description": "Statut de l'API et healthcheck."},
    {"name": "Predictions", "description": "Prédictions de prix journalier."},
]

# ---------------------------------------------------------------------------
# Types énumérés (valeurs issues de get_around_pricing_project.csv)
# ---------------------------------------------------------------------------
FuelType = Literal["diesel", "petrol", "hybrid_petrol", "electro"]
PaintColor = Literal[
    "beige", "black", "blue", "brown", "green",
    "grey", "orange", "red", "silver", "white",
]
CarType = Literal[
    "convertible", "coupe", "estate", "hatchback",
    "sedan", "subcompact", "suv", "van",
]
ModelKey = Literal[
    "Alfa Romeo", "Audi", "BMW", "Citroën", "Ferrari", "Fiat", "Ford",
    "Honda", "KIA Motors", "Lamborghini", "Lexus", "Maserati", "Mazda",
    "Mercedes", "Mini", "Mitsubishi", "Nissan", "Opel", "PGO", "Peugeot",
    "Porsche", "Renault", "SEAT", "Subaru", "Suzuki", "Toyota",
    "Volkswagen", "Yamaha",
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

    # Ordre de résolution du modèle :
    # 1. Model Registry via alias  (models:/<name>@<alias>)  → prod recommandée
    # 2. Run ID                    (runs:/<run_id>/<artifact>) → fallback / dev
    loaded = None
    logged_model: str | None = None
    last_error: Exception | None = None

    if MLFLOW_REGISTERED_MODEL and MLFLOW_MODEL_ALIAS:
        logged_model = f"models:/{MLFLOW_REGISTERED_MODEL}@{MLFLOW_MODEL_ALIAS}"
        try:
            loaded = mlflow.pyfunc.load_model(logged_model)
        except Exception as exc:
            last_error = exc
            print(
                f"Chargement via Registry impossible ({logged_model}) : "
                f"{type(exc).__name__} — {exc}. Tentative via MLFLOW_RUN_ID…"
            )
            loaded = None

    if loaded is None:
        rid = (MLFLOW_RUN_ID or "").strip()
        if not rid:
            raise RuntimeError(
                "Impossible de charger le modèle : "
                "définissez (MLFLOW_REGISTERED_MODEL + MLFLOW_MODEL_ALIAS) "
                "ou MLFLOW_RUN_ID dans .env."
                + (f" Dernière erreur Registry : {last_error}" if last_error else "")
            )
        logged_model = f"runs:/{rid}/{MLFLOW_ARTIFACT_PATH}"
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
    summary="Prédiction du prix journalier (€/jour) d'un véhicule GetAround.",
    description=description,
    version="1.0.0",
    contact={
        "name": "ZerphirosX",
        "url": "https://huggingface.co/ZerphirosX",
    },
    openapi_tags=tags_metadata,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    swagger_ui_parameters={
        "defaultModelsExpandDepth": 2,
        "docExpansion": "list",
        "displayRequestDuration": True,
        "filter": True,
        "tryItOutEnabled": True,
        "persistAuthorization": True,
    },
    lifespan=lifespan,
)


class Vehicle(BaseModel):
    """Caractéristiques d'un véhicule GetAround (ordre aligné sur `ALL_FEATURES`)."""

    model_key: ModelKey = Field(..., description="Marque du véhicule.", examples=["Citroën"])
    fuel: FuelType = Field(..., description="Type de carburant.", examples=["diesel"])
    paint_color: PaintColor = Field(..., description="Couleur de la carrosserie.", examples=["black"])
    car_type: CarType = Field(..., description="Type de carrosserie.", examples=["convertible"])
    mileage: int = Field(..., ge=0, description="Kilométrage (km).", examples=[140411])
    engine_power: int = Field(..., ge=0, description="Puissance moteur (ch).", examples=[100])
    private_parking_available: bool = Field(..., description="Parking privé disponible.", examples=[True])
    has_gps: bool = Field(..., description="GPS embarqué.", examples=[True])
    has_air_conditioning: bool = Field(..., description="Climatisation.", examples=[False])
    automatic_car: bool = Field(..., description="Boîte automatique.", examples=[False])
    has_getaround_connect: bool = Field(..., description="Équipé GetAround Connect.", examples=[True])
    has_speed_regulator: bool = Field(..., description="Régulateur de vitesse.", examples=[True])
    winter_tires: bool = Field(..., description="Pneus hiver.", examples=[True])


class PredictBody(BaseModel):
    """Corps attendu : clé `input` contenant une ou plusieurs lignes véhicule."""

    input: List[Vehicle] = Field(
        ...,
        min_length=1,
        description="Liste d'un ou plusieurs véhicules (mode batch supporté).",
    )


# ---------------------------------------------------------------------------
# Exemples nommés (affichés dans le dropdown "Try it out" de Swagger)
# ---------------------------------------------------------------------------
_EXAMPLE_CITADINE = {
    "model_key": "Peugeot",
    "fuel": "diesel",
    "paint_color": "white",
    "car_type": "hatchback",
    "mileage": 85000,
    "engine_power": 90,
    "private_parking_available": True,
    "has_gps": True,
    "has_air_conditioning": True,
    "automatic_car": False,
    "has_getaround_connect": False,
    "has_speed_regulator": True,
    "winter_tires": False,
}

_EXAMPLE_SUV_HYBRIDE = {
    "model_key": "Mercedes",
    "fuel": "hybrid_petrol",
    "paint_color": "black",
    "car_type": "suv",
    "mileage": 32000,
    "engine_power": 210,
    "private_parking_available": True,
    "has_gps": True,
    "has_air_conditioning": True,
    "automatic_car": True,
    "has_getaround_connect": True,
    "has_speed_regulator": True,
    "winter_tires": True,
}

_EXAMPLE_SPORTIVE = {
    "model_key": "Porsche",
    "fuel": "petrol",
    "paint_color": "red",
    "car_type": "coupe",
    "mileage": 18000,
    "engine_power": 380,
    "private_parking_available": True,
    "has_gps": True,
    "has_air_conditioning": True,
    "automatic_car": True,
    "has_getaround_connect": True,
    "has_speed_regulator": True,
    "winter_tires": False,
}

PREDICT_EXAMPLES = {
    "citadine_diesel": {
        "summary": "Citadine diesel économique (Peugeot hatchback)",
        "description": "Cas courant : petite voiture urbaine à prix modéré.",
        "value": {"input": [_EXAMPLE_CITADINE]},
    },
    "suv_hybride": {
        "summary": "SUV hybride premium (Mercedes)",
        "description": "Segment premium, équipements complets, kilométrage faible.",
        "value": {"input": [_EXAMPLE_SUV_HYBRIDE]},
    },
    "sportive_essence": {
        "summary": "Sportive essence haute puissance (Porsche coupé)",
        "description": "Véhicule haut de gamme, moteur puissant.",
        "value": {"input": [_EXAMPLE_SPORTIVE]},
    },
    "batch_3_vehicules": {
        "summary": "Prédiction en batch (3 véhicules)",
        "description": "Envoie plusieurs véhicules en une seule requête.",
        "value": {
            "input": [_EXAMPLE_CITADINE, _EXAMPLE_SUV_HYBRIDE, _EXAMPLE_SPORTIVE]
        },
    },
}


class PredictResponse(BaseModel):
    """Réponse de `/predict` : un prix par ligne d'entrée."""

    prediction: List[float] = Field(
        ...,
        description="Prix journaliers prédits (€/jour), arrondis à 2 décimales.",
        examples=[[145.23, 198.70]],
    )


class HealthResponse(BaseModel):
    """Réponse de `/` : statut de l'API et pointeurs utiles."""

    message: str = Field(..., examples=["Welcome to the GetAround Pricing API"])
    status: Literal["operational", "degraded"] = Field(..., examples=["operational"])
    model_loaded: bool = Field(..., examples=[True])
    docs: str = Field(..., examples=["/docs"])
    predict: str = Field(..., examples=["/predict"])


class ErrorDetail(BaseModel):
    """Format standard d'erreur renvoyé par FastAPI."""

    detail: str = Field(..., examples=["Le champ 'input' ne doit pas être vide."])


@app.get(
    "/",
    tags=["General"],
    summary="Statut de l'API",
    response_model=HealthResponse,
    responses={200: {"description": "L'API est joignable."}},
)
async def root(request: Request) -> HealthResponse:
    """Healthcheck minimal : indique si le modèle MLflow est chargé."""
    model = _PYFUNC_MODEL.get("model") or getattr(
        request.app.state, "pricing_model", None
    )
    return HealthResponse(
        message="Welcome to the GetAround Pricing API",
        status="operational" if model is not None else "degraded",
        model_loaded=model is not None,
        docs="/docs",
        predict="/predict",
    )


@app.post(
    "/predict",
    tags=["Predictions"],
    summary="Prédire le prix journalier",
    response_model=PredictResponse,
    responses={
        422: {"model": ErrorDetail, "description": "Payload invalide (types ou valeurs)."},
        503: {"model": ErrorDetail, "description": "Modèle MLflow non chargé."},
    },
)
async def predict(
    request: Request,
    body: PredictBody = Body(..., openapi_examples=PREDICT_EXAMPLES),
) -> PredictResponse:
    """
    Prédit le prix journalier (€/jour) pour un ou plusieurs véhicules.

    Chaque élément de `input` est un objet **`Vehicle`** avec 13 champs nommés
    (voir le schéma plus bas). Les champs catégoriels (`model_key`, `fuel`,
    `paint_color`, `car_type`) sont contraints aux valeurs apprises par le modèle.
    """
    model = _PYFUNC_MODEL.get("model") or getattr(
        request.app.state, "pricing_model", None
    )
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Modèle non chargé. Vérifiez les logs au démarrage.",
        )

    input_df = pd.DataFrame([v.model_dump() for v in body.input])
    # Le pipeline MLflow est entraîné avec des int64 pour les colonnes numériques
    # et les booléens encodés en 0/1 : on normalise avant predict().
    for col in ALL_FEATURES[6:]:
        input_df[col] = input_df[col].astype(int)
    for col in ("mileage", "engine_power"):
        input_df[col] = input_df[col].astype(int)

    preds = model.predict(input_df)
    return PredictResponse(prediction=[round(float(p), 2) for p in preds])


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
