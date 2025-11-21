# #!/bin/bash

# # --- Script de Configuration Automatisé et Portable pour RAG PDF App ---

# echo "🚀 Démarrage de l'installation et de la configuration du RAG PDF App."
# echo "-------------------------------------------------------------------"

# # --- CONFIG ---
# PYTHON_VENV="/opt/homebrew/bin/python3"

# VENV_NAME=".venv"

# # 1. Installer le Dépôt
# if [ ! -f "requirements.txt" ]; then
#     echo "ERREUR: Veuillez exécuter ce script depuis le répertoire RAG_app/."
#     exit 1
# fi

# # 2. Créer l'Environnement Virtuel
# if [ ! -d ".venv" ]; then
#     echo "🧠 Création de l'environnement virtuel 'venv'..."
#     python3 -m venv .venv
# fi

# # 3. Activation de l'Environnement Virtuel (Détection OS)
# echo "💻 Détection du système d'exploitation pour l'activation du venv..."
# if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
#     # Windows (exécuté via Git Bash/Cygwin)
#     VENV_ACTIVATE=".venv/Scripts/activate"
# elif [[ "$OSTYPE" == "linux-gnu" || "$OSTYPE" == "darwin"* ]]; then
#     # Linux ou macOS
#     VENV_ACTIVATE=".venv/bin/activate"
# else
#     echo "⚠️ Système d'exploitation non reconnu. Utilisation de l'activation par défaut (Linux/macOS)."
#     VENV_ACTIVATE=".venv/bin/activate"
# fi

# source "$VENV_ACTIVATE"
# echo "✅ Environnement virtuel activé : $VENV_ACTIVATE"

# # 4. Installer les Dépendances
# echo "📦 Installation des dépendances Python (voir requirements.txt)..."
# pip install -r requirements.txt

# # 5. Créer les Dossiers Requis et les fichiers .gitkeep
# echo "📂 Création des dossiers 'db', 'data', 'logs'..."
# mkdir -p .streamlit db data logs
# touch db/.gitkeep data/.gitkeep logs/.gitkeep

# # 6. Configuration des Secrets (Interaction Utilisateur)
# echo ""
# echo "--- 🔑 Configuration des Secrets (.streamlit/secrets.toml) ---"
# echo "Veuillez entrer vos identifiants pour configurer l'application."

# # Demander la clé OpenAI
# read -p "Entrez votre OPENAI_API_KEY (sk-...): " OPENAI_KEY

# # Demander les identifiants GitHub
# read -p "Entrez votre nom d'utilisateur GitHub (GIT_USER_NAME): " GIT_USER
# read -p "Entrez votre email GitHub (GIT_USER_EMAIL): " GIT_EMAIL
# read -p "Entrez votre Personal Access Token GitHub (GH_TOKEN): " GH_TOKEN

# # Créer le contenu du fichier secrets.toml
# SECRETS_CONTENT="
# OPENAI_API_KEY = \"$OPENAI_KEY\"

# GIT_USER_NAME = \"$GIT_USER\"
# GIT_USER_EMAIL = \"$GIT_EMAIL\"
# GH_TOKEN = \"$GH_TOKEN\"
# "

# # Écrire le fichier secrets.toml
# mkdir -p .streamlit
# echo "$SECRETS_CONTENT" > .streamlit/secrets.toml
# echo "✅ Fichier .streamlit/secrets.toml créé avec succès !"

# echo ""
# echo "--- 🎉 Installation Terminée ---"
# echo "Le script a activé l'environnement virtuel. Vous êtes prêt(e) !"
# echo "Lancez l'application avec :"
# echo ""
# echo "streamlit run rag_app.py"
# echo "-------------------------------------------------------------------"

#!/bin/bash

# --- Script de Configuration Automatisé et Portable pour RAG PDF App ---

echo "🚀 Démarrage de l'installation et de la configuration du RAG PDF App."
echo "-------------------------------------------------------------------"

# --- CONFIGURATION INITIALE ---
VENV_NAME=".venv"
PYTHON_CMD="python3"

if [ ! -f "requirements.txt" ]; then
    echo "ERREUR: Le fichier requirements.txt est introuvable. Veuillez exécuter ce script depuis le répertoire racine du projet."
    exit 1
fi

# Tenter de trouver python3.11, sinon utiliser python3 ou python
if command -v python3.11 &> /dev/null; then
    PYTHON_CMD="python3.11"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "ERREUR: Aucun exécutable Python (python3.11, python3, python) n'a été trouvé. Veuillez installer Python 3.11+."
    exit 1
fi
echo "Utilisation de l'exécutable: $PYTHON_CMD pour la création du VENV."


# 1. Vérifier et recréer l'Environnement Virtuel
if [ -d "$VENV_NAME" ]; then
    echo "🧠 Suppression de l'ancien environnement virtuel pour une reconstruction propre..."
    rm -rf "$VENV_NAME"
fi

echo "🧠 Création de l'environnement virtuel '$VENV_NAME'..."
if ! "$PYTHON_CMD" -m venv "$VENV_NAME"; then
    echo "ERREUR: Échec de la création du VENV. Vérifiez que Python 3.11+ est disponible."
    exit 1
fi
echo "✅ Environnement virtuel créé avec succès."


# 2. Activation de l'Environnement Virtuel (Détection OS)
echo "💻 Détection du système d'exploitation pour l'activation du venv..."
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
    # Windows (exécuté via Git Bash/Cygwin)
    VENV_ACTIVATE="$VENV_NAME/Scripts/activate"
    PIP_PATH="$VENV_NAME/Scripts/pip"
else 
    # Linux ou macOS
    VENV_ACTIVATE="$VENV_NAME/bin/activate"
    PIP_PATH="$VENV_NAME/bin/pip"
fi

# Activation de l'environnement DANS LE SHELL ACTUEL (requis par 'source')
source "$VENV_ACTIVATE"
echo "✅ Environnement virtuel activé : $VENV_ACTIVATE"


# 3. Installation des Dépendances (Utilisation du chemin complet du pip VENV)
echo "📦 Installation des dépendances Python (voir requirements.txt)..."

if [ -f "$PIP_PATH" ]; then
    if ! "$PIP_PATH" install -r requirements.txt; then
        echo "ERREUR: Échec de l'installation des dépendances via pip. Veuillez vérifier requirements.txt."
        exit 1
    fi
    echo "✅ Dépendances installées avec succès."
else
    echo "ERREUR: L'exécutable pip est introuvable à $PIP_PATH. La création du venv a échoué."
    exit 1
fi


# 4. Créer les Dossiers Requis et les fichiers .gitkeep
echo "📂 Création des dossiers 'db', 'data', 'logs' et '.streamlit'..."
mkdir -p .streamlit db data logs
touch db/.gitkeep data/.gitkeep logs/.gitkeep


# 5. Configuration des Secrets (Méthode Zsh/Bash compatible)
echo ""
echo "--- 🔑 Configuration des Secrets (.streamlit/secrets.toml) ---"
echo "Veuillez entrer vos identifiants pour configurer l'application."

# Utilisation de 'echo -n' suivi de 'read' pour la compatibilité maximale
echo -n "Entrez votre OPENAI_API_KEY (sk-...): "
read OPENAI_KEY

echo -n "Entrez votre nom d'utilisateur GitHub (GIT_USER_NAME): "
read GIT_USER

echo -n "Entrez votre email GitHub (GIT_USER_EMAIL): "
read GIT_EMAIL

echo -n "Entrez votre Personal Access Token GitHub (GH_TOKEN): "
read GH_TOKEN

# Créer le contenu du fichier secrets.toml
SECRETS_CONTENT="
OPENAI_API_KEY = \"$OPENAI_KEY\"

GIT_USER_NAME = \"$GIT_USER\"
GIT_USER_EMAIL = \"$GIT_EMAIL\"
GH_TOKEN = \"$GH_TOKEN\"
"

# Écrire le fichier secrets.toml
echo "$SECRETS_CONTENT" > .streamlit/secrets.toml
echo "✅ Fichier .streamlit/secrets.toml créé avec succès !"

echo ""
echo "--- 🎉 Installation Terminée ---"
echo "Le script a activé l'environnement virtuel. Vous êtes prêt(e) !"
echo "Lancez l'application avec :"
echo ""
echo "streamlit run rag_app.py"
echo "-------------------------------------------------------------------"