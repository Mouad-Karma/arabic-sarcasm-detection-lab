# ==============================================================================
# Dockerfile - TP 2 : Détection du Sarcasme en Arabe (ArSarcasm-v2)
# Auteur : Mouad KARMA - Master IASD / Ingénierie IA
# ==============================================================================

FROM python:3.10-slim

# Metadonnées
LABEL maintainer="Mouad KARMA"
LABEL description="Environnement de reproduction pour la détection du sarcasme en arabe sur ArSarcasm-v2"

# Variables d'environnement
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Installation des dépendances système requises
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Définition de l'espace de travail
WORKDIR /workspace

# Copie et installation des dépendances Python
COPY requirements.txt /workspace/
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copie du code, des notebooks et des livrables
COPY . /workspace/

# Exposition du port Jupyter
EXPOSE 8888

# Commande par défaut : Lancement de Jupyter Notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''"]
