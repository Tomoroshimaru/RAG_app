# RAG_app
RAG (GenAI)

# 🧠 RAG_app — Retrieval Augmented Generation (GenAI)

**RAG_app** est une application Streamlit simple et épurée permettant d’interroger dynamiquement le contenu de documents PDF via un moteur **RAG (Retrieval-Augmented Generation)**.  
Elle combine l’extraction de texte, l’encodage sémantique avec **SentenceTransformers**, la recherche vectorielle avec **FAISS**, et la génération de réponses contextuelles avec **OpenAI GPT-4o**.

---

## 🚀 Fonctionnalités

- 📂 **Upload Drag & Drop** de fichiers PDF  
- 🧩 **Extraction automatique** du texte et des tableaux via `pdfplumber`  
- 🧠 **Création d’embeddings** avec `sentence-transformers/all-MiniLM-L6-v2`  
- 🗃️ **Stockage vectoriel** local avec **FAISS (cosine similarity)**  
- 🔍 **Recherche sémantique** sur tous les documents indexés  
- 💬 **Génération de réponses** contextualisées via GPT-4o  
- 🌐 **Interface web interactive** avec Streamlit  
- 🧱 **Architecture modulaire et locale** : ingestion, requête et interface séparées  

---

## 🧩 Architecture du projet

RAG_app/
│
├── rag_app.py # Interface Streamlit (upload + requêtes)
│
├── db/ # Base vectorielle FAISS persistée
├── PDFs/ # Dossier des fichiers PDF chargés
│
├── requirements.txt # Dépendances Python
└── README.md # Ce fichier 😄

---

## ⚙️ Installation locale

### 1️⃣ Cloner le dépôt

git clone https://github.com/Tomoroshimaru/RAG_app.git
cd RAG_app

### 2️⃣ Créer un environnement virtuel (recommandé)

python -m venv venv
venv\Scripts\activate   # Windows
source venv/bin/activate  # macOS/Linux

3️⃣ Installer les dépendances

pip install -r requirements.txt

4️⃣ Ajouter ta clé OpenAI

Crée un fichier config.py (non versionné) :
OPENAI_API_KEY = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

▶️ Lancement de l’application

Exécute simplement :
streamlit run rag_app.py

