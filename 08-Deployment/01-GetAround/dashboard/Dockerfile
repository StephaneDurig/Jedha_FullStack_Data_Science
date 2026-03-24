FROM continuumio/miniconda3

RUN apt-get update -y
RUN apt-get install nano unzip curl -y

# Créer un utilisateur non-root (requis par Hugging Face Spaces)
RUN useradd -m -u 1000 user
USER user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Dépendances de base
RUN pip install boto3 pandas scikit-learn matplotlib seaborn plotly

# Copier les fichiers avec le bon owner
COPY --chown=user . $HOME/app

COPY requirements.txt /requirements.txt
RUN pip install -r /requirements.txt

# HF injecte PORT ; défaut 7860 si absent (évite que --server.address soit pris pour le port)
CMD ["sh", "-c", "exec streamlit run app.py --server.port ${PORT:-7860} --server.address 0.0.0.0 --server.headless true"]
