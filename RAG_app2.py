# rag_app.py (version corrigée - séparation upload/query)
import streamlit as st
import os
import pdfplumber
import textwrap
import numpy as np
import faiss
import pickle
import warnings
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import streamlit as st

# Supprimer les warnings de pdfplumber
warnings.filterwarnings("ignore", message=".*CropBox.*")

# Config
DATA_DIR = "data"
DB_DIR = "db"
MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500
TOP_K = 5

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)

st.set_page_config(page_title="RAG PDF App", page_icon="🧠", layout="centered")

# Load models
@st.cache_resource
def load_model():
    return SentenceTransformer(MODEL_NAME)

model = load_model()
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=OPENAI_API_KEY)

# -------- Helper Functions --------
def extract_text_tables(file_bytes, filename):
    """Extrait le texte et les tableaux d'un PDF"""
    paragraphs = []
    metadata = []
    
    with pdfplumber.open(file_bytes) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            if text:
                chunks = textwrap.wrap(text.strip(), width=CHUNK_SIZE)
                for chunk in chunks:
                    if chunk.strip():
                        paragraphs.append(chunk)
                        metadata.append({
                            "source": filename,
                            "page": page_num,
                            "type": "text",
                            "text": chunk
                        })
            
            tables = page.extract_tables()
            for table in tables or []:
                table_str = "\n".join(
                    [" | ".join([str(cell or "") for cell in row]) for row in table if row]
                )
                if table_str.strip():
                    paragraphs.append(table_str)
                    metadata.append({
                        "source": filename,
                        "page": page_num,
                        "type": "table",
                        "text": table_str
                    })
    
    return paragraphs, metadata

def load_or_create_index():
    """Charge l'index FAISS existant ou retourne None"""
    index_path = os.path.join(DB_DIR, "index.faiss")
    meta_path = os.path.join(DB_DIR, "metadata.pkl")
    
    if os.path.exists(index_path) and os.path.exists(meta_path):
        try:
            if os.path.getsize(index_path) > 0:
                index = faiss.read_index(index_path)
                with open(meta_path, "rb") as f:
                    metadata = pickle.load(f)
                return index, metadata
            else:
                st.warning("⚠️ Fichier index.faiss vide, création d'un nouvel index...")
                return None, []
        except Exception as e:
            st.warning(f"⚠️ Erreur lors du chargement de l'index : {e}. Création d'un nouvel index...")
            return None, []
    
    return None, []

def save_index(index, metadata):
    """Sauvegarde l'index FAISS et les métadonnées"""
    index_path = os.path.join(DB_DIR, "index.faiss")
    meta_path = os.path.join(DB_DIR, "metadata.pkl")
    
    try:
        faiss.write_index(index, index_path)
        with open(meta_path, "wb") as f:
            pickle.dump(metadata, f)
        return True
    except Exception as e:
        st.error(f"❌ Erreur lors de la sauvegarde : {e}")
        return False

# -------- Sidebar Stats --------
with st.sidebar:
    st.header("📊 Statistiques")
    
    index, metadata = load_or_create_index()
    
    if index is not None and metadata:
        st.metric("Nombre de segments", len(metadata))
        
        sources = set([m.get("source", "Unknown") for m in metadata])
        st.metric("Nombre de documents", len(sources))
        
        st.write("**Documents indexés :**")
        for source in sorted(sources):
            count = sum(1 for m in metadata if m.get("source") == source)
            st.write(f"- {source} ({count} segments)")
    else:
        st.info("Aucun document indexé pour le moment.")
    


# -------- Main Title --------
st.title("🧠 RAG sur documents PDF")

# -------- Tabs pour séparer Upload et Query --------
tab1, tab2 = st.tabs(["📤 Upload de documents", "❓ Poser une question"])

# ========== TAB 1: UPLOAD ==========
with tab1:
    st.header("📤 Ajouter des documents")
    
    # Bouton pour réinitialiser la base en haut
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("🗑️ Réinitialiser la base", key="reset_btn_tab", type="secondary"):
            index_path = os.path.join(DB_DIR, "index.faiss")
            meta_path = os.path.join(DB_DIR, "metadata.pkl")
            
            if os.path.exists(index_path):
                os.remove(index_path)
            if os.path.exists(meta_path):
                os.remove(meta_path)
            
            st.success("✅ Base réinitialisée !")
            st.rerun()
    
    uploaded_files = st.file_uploader(
        "Dépose tes fichiers PDF ici :", 
        accept_multiple_files=True, 
        type=["pdf"],
        key="pdf_uploader"
    )
    
    if st.button("🚀 Indexer les documents", key="index_btn"):
        if not uploaded_files:
            st.warning("⚠️ Veuillez d'abord sélectionner des fichiers PDF.")
        else:
            with st.spinner("📚 Traitement des fichiers PDF..."):
                corpus, new_metadata = [], []
                
                for file in uploaded_files:
                    # Sauvegarder le fichier
                    file_path = os.path.join(DATA_DIR, file.name)
                    with open(file_path, "wb") as f:
                        f.write(file.getbuffer())
                    st.success(f"✅ {file.name} enregistré.")
                    
                    # Extraire le contenu
                    paragraphs, file_metadata = extract_text_tables(file, file.name)
                    corpus.extend(paragraphs)
                    new_metadata.extend(file_metadata)
                
                if not corpus:
                    st.error("❌ Aucun contenu extrait des PDFs.")
                    st.stop()
                
                st.info(f"🧠 Création des embeddings pour {len(corpus)} segments...")
                
                # Créer les embeddings
                embeddings = model.encode(corpus, normalize_embeddings=True, show_progress_bar=False)
                embeddings = np.array(embeddings, dtype=np.float32)
                
                # Charger ou créer l'index
                existing_index, existing_metadata = load_or_create_index()
                
                if existing_index is not None:
                    # Ajouter à l'index existant
                    existing_index.add(embeddings)
                    combined_metadata = existing_metadata + new_metadata
                    index = existing_index
                else:
                    # Créer un nouvel index
                    dimension = embeddings.shape[1]
                    index = faiss.IndexFlatIP(dimension)
                    index.add(embeddings)
                    combined_metadata = new_metadata
                
                # Sauvegarder
                if save_index(index, combined_metadata):
                    st.success(f"🎉 Base FAISS mise à jour avec succès ! Total de {len(combined_metadata)} segments.")
                    st.balloons()
                else:
                    st.error("❌ Échec de la sauvegarde de l'index.")

# ========== TAB 2: QUERY ==========
with tab2:
    st.header("❓ Pose ta question")
    
    # Vérifier qu'il y a des documents indexés
    index, metadata = load_or_create_index()
    
    if index is None or len(metadata) == 0:
        st.warning("⚠️ Aucun document indexé. Veuillez d'abord uploader et indexer des documents dans l'onglet 'Upload de documents'.")
    else:
        query = st.text_input("Entrez votre question :", key="query_input")
        
        if st.button("🔍 Rechercher", key="search_btn"):
            if not query.strip():
                st.warning("⚠️ Veuillez entrer une question.")
            else:
                with st.spinner("🔍 Recherche en cours..."):
                    # Encoder la requête
                    query_emb = model.encode([query], normalize_embeddings=True)
                    query_emb = np.array(query_emb, dtype=np.float32)
                    
                    # Rechercher les segments les plus similaires
                    D, I = index.search(query_emb, k=min(TOP_K, len(metadata)))
                    
                    # Récupérer les textes
                    top_chunks = []
                    sources_info = []
                    
                    for idx, score in zip(I[0], D[0]):
                        if idx < len(metadata):
                            entry = metadata[idx]
                            text = entry.get("text", "")
                            if text:
                                top_chunks.append(text)
                                sources_info.append({
                                    "source": entry.get('source'),
                                    "page": entry.get('page'),
                                    "score": float(score),
                                    "type": entry.get('type')
                                })
                    
                    if not top_chunks:
                        st.error("❌ Aucun texte trouvé dans les métadonnées.")
                    else:
                        # Afficher les sources
                        with st.expander("📚 Sources utilisées (cliquez pour voir)"):
                            for i, source in enumerate(sources_info, 1):
                                st.write(f"**{i}.** 📄 {source['source']} (page {source['page']}, {source['type']}) - Similarité: {source['score']:.3f}")
                        
                        # Construire le prompt
                        context = "\n\n".join(top_chunks)
                        prompt = f"""Contexte extrait des documents :
{context}

Question : {query}

Réponds de manière précise et factuelle en te basant uniquement sur le contexte fourni. 
Si l'information n'est pas dans le contexte, indique-le clairement."""

                        st.info("🧠 Génération de la réponse...")
                        
                        try:
                            response = client.chat.completions.create(
                                model="gpt-4o-mini",
                                messages=[
                                    {
                                        "role": "system", 
                                        "content": "Tu es un assistant spécialisé dans l'analyse de documents PDF. Tu réponds uniquement à partir du contexte fourni."
                                    },
                                    {"role": "user", "content": prompt},
                                ],
                                temperature=0.3,
                                max_tokens=1000
                            )
                            
                            answer = response.choices[0].message.content.strip()
                            
                            st.subheader("🎯 Réponse générée :")
                            st.write(answer)
                            
                        except Exception as e:
                            st.error(f"❌ Erreur lors de la génération : {e}")
