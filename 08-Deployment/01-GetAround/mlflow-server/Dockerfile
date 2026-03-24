FROM continuumio/miniconda3

RUN apt-get update -y
RUN apt-get install nano unzip curl -y

# Installer l'AWS CLI (nécessaire pour interagir avec S3)
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
RUN unzip awscliv2.zip
RUN ./aws/install

# Créer un utilisateur non-root (requis par Hugging Face Spaces)
RUN useradd -m -u 1000 user
USER user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

COPY --chown=user . $HOME/app

COPY requirements.txt /requirements.txt
RUN pip install -r /requirements.txt

# Variables d'environnement AWS et MLflow (injectées via HF Spaces Secrets)
ENV AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID
ENV AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY
ENV BACKEND_STORE_URI=$BACKEND_STORE_URI
ENV ARTIFACT_STORE_URI=$ARTIFACT_STORE_URI

# Lancer le serveur MLflow
# HF Spaces injecte souvent $PORT ; si absent, 7860 (port HF par défaut).
# Une seule ligne shell évite que -p soit vide (sinon --host est pris comme port).
CMD ["sh", "-c", "exec mlflow server -p ${PORT:-7860} --host 0.0.0.0 --backend-store-uri \"$BACKEND_STORE_URI\" --default-artifact-root \"$ARTIFACT_STORE_URI\""]
