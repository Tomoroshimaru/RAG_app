# rag_app.py - RAG with GitHub Sync
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
MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500
TOP_K = 10

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)

# Conversation state
if "history" not in st.session_state:
    st.session_state.history = []

st.set_page_config(page_title="RAG PDF App", page_icon="🧠", layout="centered")

# Load models
@st.cache_resource
def load_model():
    return SentenceTransformer(MODEL_NAME)

model = load_model()
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=OPENAI_API_KEY)

# Helper function
def extract_text_tables(file_bytes, filename):
    """Extract text and tables from PDF."""
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
                st.write(f"• {source} ({count})")
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
        st.success("✅ Database cleared!")
        st.rerun()

# Main Title
st.title("🧠 RAG on PDF Documents")

# Tabs
tab1, tab2 = st.tabs(["❓ Ask Questions", "📤 Upload Documents"])

# TAB 1: UPLOAD
with tab2:
    st.header("📤 Add Documents")
    
    uploaded_files = st.file_uploader(
        "Drop your PDF files here:", 
        accept_multiple_files=True, 
        type=["pdf"],
        key="pdf_uploader"
    )
    
    if st.button("🚀 Index Documents", key="index_btn"):
        if not uploaded_files:
            st.warning("⚠️ Please select PDF files first.")
        else:
            with st.spinner("📚 Processing PDFs..."):
                corpus, new_metadata = [], []
                
                for file in uploaded_files:
                    # Save file
                    file_path = os.path.join(DATA_DIR, file.name)
                    with open(file_path, "wb") as f:
                        f.write(file.getbuffer())
                    st.success(f"✅ {file.name} saved.")
                    
                    # Extract content
                    paragraphs, file_metadata = extract_text_tables(file, file.name)
                    corpus.extend(paragraphs)
                    new_metadata.extend(file_metadata)
                
                if not corpus:
                    st.error("❌ No content extracted from PDFs.")
                    st.stop()
                
                st.info(f"🧠 Creating embeddings for {len(corpus)} segments...")
                
                # Create embeddings
                embeddings = model.encode(corpus, normalize_embeddings=True, show_progress_bar=False)
                embeddings = np.array(embeddings, dtype=np.float32)
                
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

# TAB 2: QUERY
with tab1:
    st.header("Explore your Documents")
    
    # Check if documents are indexed
    index, metadata = load_index()
    
    if index is None or len(metadata) == 0:
        st.warning("⚠️ No documents indexed. Please upload and index documents first.")
    else:
        # Display conversation history
        if "history" in st.session_state and st.session_state.history:
            st.write("### Conversation")
            for item in st.session_state.history:
                st.markdown(f"**🧑‍💻 Question:** {item['query']}")
                st.markdown(f"**🤖 Answer:** {item['answer']}")
                st.markdown("---")

        # User input
        query = st.text_input("Enter your question:", key="query_input")
        
        if st.button("🔍 Search", key="search_btn"):
            if not query.strip():
                st.warning("⚠️ Please enter a question.")
            else:
                with st.spinner("🔍 Searching..."):
                    # Encode query
                    query_emb = model.encode([query], normalize_embeddings=True)
                    query_emb = np.array(query_emb, dtype=np.float32)
                    
                    # Search
                    D, I = index.search(query_emb, k=min(TOP_K, len(metadata)))
                    
                    # Retrieve texts
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
                        st.error("❌ No text found.")
                    else:
                        # Show sources
                        with st.expander("📚 Sources used (click to view)"):
                            for i, source in enumerate(sources_info, 1):
                                st.write(f"**{i}.** 📄 {source['source']} (page {source['page']}, {source['type']}) - Score: {source['score']:.3f}")
                        
                        # Build prompt
                        context = "\n\n".join(top_chunks)
                        prompt = f"""Context from documents:
{context}

Question: {query}

Answer precisely based only on the context provided. If the information is not in the context, say so clearly."""

                        st.info("🧠 Generating answer...")
                        
                        try:
                            response = client.chat.completions.create(
                                model="gpt-4o-mini",
                                messages=[
                                    {
                                        "role": "system", 
                                        "content": "You are an assistant specialized in analyzing PDF documents. Answer only from the provided context."
                                    },
                                    {"role": "user", "content": prompt},
                                ],
                                temperature=0.3,
                                max_tokens=1000
                            )
                            
                            answer = response.choices[0].message.content.strip()
                            
                            st.subheader("🎯 Generated Answer:")
                            st.write(answer)

                            st.session_state.history.append({
                                "query": query,
                                "answer": answer
                            })
                            
                        except Exception as e:
                            st.error(f"❌ Generation error: {e}")