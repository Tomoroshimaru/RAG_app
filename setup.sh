#!/bin/bash

# --- Script de Configuration Automatisé et Portable pour RAG PDF App ---

echo "🚀 Démarrage de l'installation et de la configuration du RAG PDF App."
echo "-------------------------------------------------------------------"

# 1. Installer le Dépôt
if [ ! -f "requirements.txt" ]; then
    echo "ERREUR: Veuillez exécuter ce script depuis le répertoire RAG_app/."
    exit 1
fi

# 2. Créer l'Environnement Virtuel
if [ ! -d ".venv" ]; then
    echo "🧠 Création de l'environnement virtuel 'venv'..."
    python3 -m venv .venv
fi

# 3. Activation de l'Environnement Virtuel (Détection OS)
echo "💻 Détection du système d'exploitation pour l'activation du venv..."
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
    # Windows (exécuté via Git Bash/Cygwin)
    VENV_ACTIVATE=".venv/Scripts/activate"
elif [[ "$OSTYPE" == "linux-gnu" || "$OSTYPE" == "darwin"* ]]; then
    # Linux ou macOS
    VENV_ACTIVATE=".venv/bin/activate"
else
    echo "⚠️ Système d'exploitation non reconnu. Utilisation de l'activation par défaut (Linux/macOS)."
    VENV_ACTIVATE=".venv/bin/activate"
fi

source "$VENV_ACTIVATE"
echo "✅ Environnement virtuel activé : $VENV_ACTIVATE"

# 4. Installer les Dépendances
echo "📦 Installation des dépendances Python (voir requirements.txt)..."
pip install -r requirements.txt

# 5. Créer les Dossiers Requis et les fichiers .gitkeep
echo "📂 Création des dossiers 'db', 'data', 'logs'..."
mkdir -p .streamlit db data logs
touch db/.gitkeep data/.gitkeep logs/.gitkeep

# 6. Configuration des Secrets (Interaction Utilisateur)
echo ""
echo "--- 🔑 Configuration des Secrets (.streamlit/secrets.toml) ---"
echo "Veuillez entrer vos identifiants pour configurer l'application."

# Demander la clé OpenAI
read -p "Entrez votre OPENAI_API_KEY (sk-...): " OPENAI_KEY

# Demander les identifiants GitHub
read -p "Entrez votre nom d'utilisateur GitHub (GIT_USER_NAME): " GIT_USER
read -p "Entrez votre email GitHub (GIT_USER_EMAIL): " GIT_EMAIL
read -p "Entrez votre Personal Access Token GitHub (GH_TOKEN): " GH_TOKEN

# Créer le contenu du fichier secrets.toml
SECRETS_CONTENT="
OPENAI_API_KEY = \"$OPENAI_KEY\"

GIT_USER_NAME = \"$GIT_USER\"
GIT_USER_EMAIL = \"$GIT_EMAIL\"
GH_TOKEN = \"$GH_TOKEN\"
"

# Écrire le fichier secrets.toml
mkdir -p .streamlit
echo "$SECRETS_CONTENT" > .streamlit/secrets.toml
echo "✅ Fichier .streamlit/secrets.toml créé avec succès !"

echo ""
echo "--- 🎉 Installation Terminée ---"
echo "Le script a activé l'environnement virtuel. Vous êtes prêt(e) !"
echo "Lancez l'application avec :"
echo ""
echo "streamlit run rag_app.py"
echo "-------------------------------------------------------------------"