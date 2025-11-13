# rag_app.py - RAG with GitHub Sync (Optimized)
import streamlit as st
import os
import pdfplumber
import textwrap
import numpy as np
import faiss
import warnings
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from faiss_manager import load_index, save_index, clear_index, push_to_github, pull_from_github

# Suppress warnings
warnings.filterwarnings("ignore", message=".*CropBox.*")

# Config
DATA_DIR = "data"
DB_DIR = "db"
MODEL_NAME = "intfloat/multilingual-e5-base"
CHUNK_SIZE = 500
TOP_K = 10

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)

st.set_page_config(page_title="RAG PDF App", page_icon="🧠", layout="centered")

# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = []
if "last_query" not in st.session_state:
    st.session_state.last_query = ""

# Load models
@st.cache_resource
def load_model():
    return SentenceTransformer(MODEL_NAME)

model = load_model()
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=OPENAI_API_KEY)

# Helper functions
def extract_text_tables(file_bytes, filename):
    """Extract text and tables from PDF."""
    paragraphs = []
    metadata = []
    
    with pdfplumber.open(file_bytes) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            # Extract text
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
            
            # Extract tables
            tables = page.extract_tables()
            if tables:
                for table in tables:
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

def encode_texts(texts, prefix="passage"):
    """Encode texts with E5 model prefix."""
    formatted = [f"{prefix}: {text}" for text in texts]
    embeddings = model.encode(formatted, normalize_embeddings=True, show_progress_bar=False)
    return np.array(embeddings, dtype=np.float32)

def search_similar(query, index, metadata, top_k=TOP_K):
    """Search for similar documents."""
    # Encode query with E5 prefix
    query_emb = encode_texts([query], prefix="query")
    
    # Search
    D, I = index.search(query_emb, k=min(top_k, len(metadata)))
    
    # Retrieve results
    results = []
    for idx, score in zip(I[0], D[0]):
        if idx < len(metadata):
            entry = metadata[idx]
            results.append({
                "text": entry.get("text", ""),
                "source": entry.get("source", "Unknown"),
                "page": entry.get("page", 0),
                "type": entry.get("type", "text"),
                "score": float(score)
            })
    
    return results

# Sidebar
with st.sidebar:
    st.header("📊 Database Management")
    
    # Pull from GitHub
    if st.button("⬇️ Pull from GitHub", use_container_width=True):
        with st.spinner("Pulling..."):
            success, message = pull_from_github()
            if success:
                st.success(message)
                st.rerun()
            else:
                st.warning(message)
    
    # Stats
    index, metadata = load_index()
    
    if index is not None and metadata:
        st.metric("Segments", len(metadata))
        sources = set([m.get("source", "Unknown") for m in metadata])
        st.metric("Documents", len(sources))
        
        with st.expander("📄 Indexed Documents"):
            for source in sorted(sources):
                count = sum(1 for m in metadata if m.get("source") == source)
                st.write(f"• {source} ({count} segments)")
    else:
        st.info("No documents indexed yet.")
    
    st.divider()
    
    # Push to GitHub
    if st.button("⬆️ Push to GitHub", type="primary", use_container_width=True):
        with st.spinner("Pushing..."):
            success, message = push_to_github()
            if success:
                st.success(message)
            else:
                st.warning(message)
    
    # Clear database
    if st.button("🗑️ Clear Database", use_container_width=True):
        clear_index()
        st.session_state.history = []  # Clear history too
        st.success("✅ Database cleared!")
        st.rerun()
    
    # Clear conversation
    if st.button("🧹 Clear Conversation", use_container_width=True):
        st.session_state.history = []
        st.session_state.last_query = ""
        st.success("✅ Conversation cleared!")
        st.rerun()

# Main Title
st.title("🧠 RAG on PDF Documents")

# Tabs
tab1, tab2 = st.tabs(["Ask questions", "Upload documents"])

# TAB 2: UPLOAD
with tab2:
    st.header("📤 Add Documents")
    
    uploaded_files = st.file_uploader(
        "Drop your PDF files here:", 
        accept_multiple_files=True, 
        type=["pdf"],
        key="pdf_uploader"
    )
    
    if st.button("🚀 Index Documents", key="index_btn", disabled=not uploaded_files):
        with st.spinner("📚 Processing PDFs..."):
            corpus, new_metadata = [], []
            progress_bar = st.progress(0)
            
            for i, file in enumerate(uploaded_files):
                # Update progress
                progress_bar.progress((i + 1) / len(uploaded_files))
                
                # Save file
                file_path = os.path.join(DATA_DIR, file.name)
                with open(file_path, "wb") as f:
                    f.write(file.getbuffer())
                
                # Extract content
                paragraphs, file_metadata = extract_text_tables(file, file.name)
                corpus.extend(paragraphs)
                new_metadata.extend(file_metadata)
                
                st.success(f"✅ {file.name} processed ({len(paragraphs)} segments)")
            
            progress_bar.empty()
            
            if not corpus:
                st.error("❌ No content extracted from PDFs.")
                st.stop()
            
            st.info(f"🧠 Creating embeddings for {len(corpus)} segments...")
            
            # Create embeddings with E5 prefix
            embeddings = encode_texts(corpus, prefix="passage")
            
            # Load or create index
            existing_index, existing_metadata = load_index()
            
            if existing_index is not None:
                # Add to existing index
                existing_index.add(embeddings)
                combined_metadata = existing_metadata + new_metadata
                index = existing_index
            else:
                # Create new index
                dimension = embeddings.shape[1]
                index = faiss.IndexFlatIP(dimension)
                index.add(embeddings)
                combined_metadata = new_metadata
            
            # Save
            if save_index(index, combined_metadata):
                st.success(f"🎉 Database updated! Total: {len(combined_metadata)} segments.")
                st.info("💡 Don't forget to push to GitHub using the sidebar button!")
                st.balloons()
            else:
                st.error("❌ Failed to save index.")

# TAB 1: QUERY
with tab1:
    st.header("❓ Explore Your Documents")
    
    # Check if documents are indexed
    index, metadata = load_index()
    
    if index is None or len(metadata) == 0:
        st.warning("⚠️ No documents indexed. Please upload and index documents first in the 'Upload Documents' tab.")
    else:
        # Display conversation history
        if st.session_state.history:
            with st.expander("📜 Conversation History", expanded=False):
                for i, item in enumerate(reversed(st.session_state.history[-5:]), 1):
                    st.markdown(f"**Q{i}:** {item['query']}")
                    st.markdown(f"**A{i}:** {item['answer'][:200]}...")
                    st.markdown("---")
        
        # User input
        query = st.text_input(
            "Enter your question:", 
            key="query_input",
            placeholder="e.g., What are the main findings in the document?"
        )
        
        col1, col2 = st.columns([3, 1])
        with col1:
            search_btn = st.button("🔍 Search", key="search_btn", type="primary", use_container_width=True)
        with col2:
            clear_btn = st.button("🔄 Clear", key="clear_query_btn", use_container_width=True)
        
        if clear_btn:
            st.session_state.last_query = ""
            st.rerun()
        
        if search_btn:
            if not query.strip():
                st.warning("⚠️ Please enter a question.")
            else:
                # Avoid duplicate searches
                if query == st.session_state.last_query:
                    st.info("💡 This question was already asked. Check the history above.")
                    st.stop()
                
                st.session_state.last_query = query
                
                with st.spinner("🔍 Searching..."):
                    # Search similar documents
                    results = search_similar(query, index, metadata, top_k=TOP_K)
                    
                    if not results:
                        st.error("❌ No relevant information found.")
                    else:
                        # Show sources
                        with st.expander(f"📚 Top {len(results)} Sources (click to view)"):
                            for i, result in enumerate(results, 1):
                                st.write(
                                    f"**{i}.** 📄 {result['source']} "
                                    f"(page {result['page']}, {result['type']}) - "
                                    f"Score: {result['score']:.3f}"
                                )
                        
                        # Build context
                        context = "\n\n".join([r["text"] for r in results])
                        
                        # Build prompt with conversation history
                        history_context = ""
                        if st.session_state.history:
                            recent = st.session_state.history[-3:]
                            history_context = "Recent conversation:\n" + "\n".join(
                                [f"Q: {h['query']}\nA: {h['answer'][:150]}..." for h in recent]
                            ) + "\n\n"
                        
                        prompt = f"""{history_context}Context from documents:
{context}

Question: {query}

Answer precisely based on the context provided. If the information is not in the context, say so clearly. Be concise and factual."""

                        st.info("🧠 Generating answer...")
                        
                        try:
                            response = client.chat.completions.create(
                                model="gpt-4o-mini",
                                messages=[
                                    {
                                        "role": "system", 
                                        "content": "You are an assistant specialized in analyzing PDF documents. Answer only from the provided context. Be concise and precise."
                                    },
                                    {"role": "user", "content": prompt},
                                ],
                                temperature=0.3,
                                max_tokens=800
                            )
                            
                            answer = response.choices[0].message.content.strip()
                            
                            st.subheader("🎯 Generated Answer:")
                            st.write(answer)
                            
                            # Save to history
                            st.session_state.history.append({
                                "query": query,
                                "answer": answer
                            })
                            
                        except Exception as e:
                            st.error(f"❌ Generation error: {e}")