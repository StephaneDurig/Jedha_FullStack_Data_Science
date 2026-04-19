FROM continuumio/miniconda3

RUN apt-get update -y
RUN apt-get install nano unzip curl -y

# Même famille majeure que le notebook d’entraînement (évite l’avertissement MLflow Python 3.12 vs 3.13)
RUN conda install -y python=3.12 pip && conda clean -afy

# Créer un utilisateur non-root (requis par Hugging Face Spaces)
RUN useradd -m -u 1000 user
USER user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Copier les fichiers avec le bon owner
COPY --chown=user . $HOME/app

COPY requirements.txt /requirements.txt
RUN pip install -r /requirements.txt

# Variables d'environnement MLflow (injectées via HF Spaces Secrets)
# Chargement modèle : priorité au Model Registry (alias), sinon fallback sur RUN_ID.
ENV MLFLOW_TRACKING_URI=$MLFLOW_TRACKING_URI
ENV MLFLOW_REGISTERED_MODEL=$MLFLOW_REGISTERED_MODEL
ENV MLFLOW_MODEL_ALIAS=$MLFLOW_MODEL_ALIAS
ENV MLFLOW_ARTIFACT_PATH=$MLFLOW_ARTIFACT_PATH
ENV MLFLOW_RUN_ID=$MLFLOW_RUN_ID

# HF injecte souvent PORT ; défaut 7860 si absent (évite --port sans valeur).
CMD ["sh", "-c", "exec fastapi run app.py --port ${PORT:-7860}"]
