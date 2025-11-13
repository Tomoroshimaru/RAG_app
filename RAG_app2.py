# rag_app.py - RAG with GitHub Sync + Cross-Lingual Multi-Query
import streamlit as st
import os
import pdfplumber
import textwrap
import numpy as np
import faiss
import warnings
import json
from langdetect import detect
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
TOP_K_PER_QUERY = 5  # Per query variant
TOP_K_FINAL = 15     # Final chunks after reranking
ENABLE_REWRITING = True
ENABLE_TRANSLATION = True

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

def detect_language(query):
    """Detect query language (FR or EN)."""
    try:
        lang = detect(query)
        return "FR" if lang.startswith("fr") else "EN"
    except:
        return "EN"  # Default to EN if detection fails

def rewrite_query(query, lang):
    """Generate 2 reformulations in the detected language."""
    if not ENABLE_REWRITING:
        return [query]
    
    lang_instruction = "French" if lang == "FR" else "English"
    
    rewrite_prompt = f"""You are a query rewriting specialist. Generate 2 reformulations of this query in {lang_instruction}:

- Use only synonyms or semantic equivalents
- Keep the exact same meaning
- No generalization
- Be concise

Original query:
{query}

Respond in JSON format:
{{
  "q1": "<reformulation_1>",
  "q2": "<reformulation_2>"
}}
"""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": rewrite_prompt}],
            temperature=0.3,
            max_tokens=200
        )
        
        content = response.choices[0].message.content.strip()
        # Clean markdown blocks
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:].strip()
        
        rewrites = json.loads(content)
        return [query, rewrites["q1"], rewrites["q2"]]
    
    except Exception as e:
        st.warning(f"⚠️ Rewriting failed: {e}. Using original query only.")
        return [query]

def translate_queries(queries, source_lang):
    """Translate 3 queries to the other language."""
    if not ENABLE_TRANSLATION:
        return []
    
    target_lang = "English" if source_lang == "FR" else "French"
    
    translate_prompt = f"""Translate these 3 queries to {target_lang}. Keep the exact same meaning.

Queries:
1. {queries[0]}
2. {queries[1]}
3. {queries[2]}

Respond in JSON format:
{{
  "t1": "<translation_1>",
  "t2": "<translation_2>",
  "t3": "<translation_3>"
}}
"""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": translate_prompt}],
            temperature=0.3,
            max_tokens=300
        )
        
        content = response.choices[0].message.content.strip()
        # Clean markdown blocks
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:].strip()
        
        translations = json.loads(content)
        return [translations["t1"], translations["t2"], translations["t3"]]
    
    except Exception as e:
        st.warning(f"⚠️ Translation failed: {e}. Using source language only.")
        return []

def search_and_rerank(queries, index, metadata, top_k_per_query=TOP_K_PER_QUERY, top_k_final=TOP_K_FINAL):
    """Search with 6 queries, deduplicate, rerank by max score."""
    # Dictionary to store: {chunk_text: (max_score, metadata)}
    chunk_scores = {}
    
    for query_text in queries:
        # Encode and search
        query_emb = encode_texts([query_text], prefix="query")
        D, I = index.search(query_emb, k=min(top_k_per_query, len(metadata)))
        
        # Collect results
        for idx, score in zip(I[0], D[0]):
            if idx < len(metadata):
                entry = metadata[idx]
                text = entry.get("text", "")
                
                if text:
                    # Keep max score for each unique chunk
                    if text not in chunk_scores or score > chunk_scores[text][0]:
                        chunk_scores[text] = (
                            float(score),
                            {
                                "source": entry.get("source", "Unknown"),
                                "page": entry.get("page", 0),
                                "type": entry.get("type", "text"),
                            }
                        )
    
    # Sort by score (descending) and take top K
    ranked = sorted(chunk_scores.items(), key=lambda x: x[1][0], reverse=True)
    
    # Format results
    results = []
    for text, (score, meta) in ranked[:top_k_final]:
        results.append({
            "text": text,
            "score": score,
            "source": meta["source"],
            "page": meta["page"],
            "type": meta["type"]
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
    
    # Settings
    st.subheader("⚙️ Search Settings")
    ENABLE_REWRITING = st.checkbox("Enable Query Rewriting", value=True, 
                                    help="Generate query variants for better recall")
    ENABLE_TRANSLATION = st.checkbox("Enable Cross-Lingual Search", value=True,
                                     help="Search in both FR and EN (6 queries total)")
    
    if ENABLE_TRANSLATION:
        st.info("🌍 Will search in FR + EN (6 queries)")
    elif ENABLE_REWRITING:
        st.info("🔄 Will search with 3 variants")
    else:
        st.info("📝 Single query search")
    
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
        st.session_state.history = []
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
tab1, tab2 = st.tabs(["❓ Ask Questions", "📤 Upload Documents"])

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
                progress_bar.progress((i + 1) / len(uploaded_files))
                
                file_path = os.path.join(DATA_DIR, file.name)
                with open(file_path, "wb") as f:
                    f.write(file.getbuffer())
                
                paragraphs, file_metadata = extract_text_tables(file, file.name)
                corpus.extend(paragraphs)
                new_metadata.extend(file_metadata)
                
                st.success(f"✅ {file.name} processed ({len(paragraphs)} segments)")
            
            progress_bar.empty()
            
            if not corpus:
                st.error("❌ No content extracted from PDFs.")
                st.stop()
            
            st.info(f"🧠 Creating embeddings for {len(corpus)} segments...")
            
            embeddings = encode_texts(corpus, prefix="passage")
            
            existing_index, existing_metadata = load_index()
            
            if existing_index is not None:
                existing_index.add(embeddings)
                combined_metadata = existing_metadata + new_metadata
                index = existing_index
            else:
                dimension = embeddings.shape[1]
                index = faiss.IndexFlatIP(dimension)
                index.add(embeddings)
                combined_metadata = new_metadata
            
            if save_index(index, combined_metadata):
                st.success(f"🎉 Database updated! Total: {len(combined_metadata)} segments.")
                st.info("💡 Don't forget to push to GitHub using the sidebar button!")
                st.balloons()
            else:
                st.error("❌ Failed to save index.")

# TAB 1: QUERY
with tab1:
    st.header("❓ Explore Your Documents")
    
    index, metadata = load_index()
    
    if index is None or len(metadata) == 0:
        st.warning("⚠️ No documents indexed. Please upload and index documents first.")
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
            placeholder="Ask in French or English..."
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
                if query == st.session_state.last_query:
                    st.info("💡 This question was already asked. Check the history above.")
                    st.stop()
                
                st.session_state.last_query = query
                
                with st.spinner("🔍 Processing query..."):
                    # Step 1: Detect language
                    detected_lang = detect_language(query)
                    st.info(f"🌐 Detected language: {detected_lang}")
                    
                    all_queries = []
                    
                    # Step 2: Generate rewrites in source language
                    if ENABLE_REWRITING:
                        with st.spinner(f"📝 Generating {detected_lang} variants..."):
                            source_queries = rewrite_query(query, detected_lang)
                            all_queries.extend(source_queries)
                            
                            if len(source_queries) > 1:
                                with st.expander(f"🔄 {detected_lang} Query Variants"):
                                    for i, q in enumerate(source_queries, 1):
                                        st.write(f"**{i}.** {q}")
                    else:
                        all_queries = [query]
                    
                    # Step 3: Translate to other language
                    if ENABLE_TRANSLATION and len(all_queries) == 3:
                        target_lang = "EN" if detected_lang == "FR" else "FR"
                        with st.spinner(f"🌍 Translating to {target_lang}..."):
                            translated = translate_queries(all_queries, detected_lang)
                            
                            if translated:
                                all_queries.extend(translated)
                                with st.expander(f"🌍 {target_lang} Translations"):
                                    for i, q in enumerate(translated, 1):
                                        st.write(f"**{i}.** {q}")
                    
                    st.info(f"🔍 Searching with {len(all_queries)} query variants...")
                    
                    # Step 4: Search and rerank
                    with st.spinner("🔍 Searching documents..."):
                        results = search_and_rerank(
                            all_queries, 
                            index, 
                            metadata,
                            top_k_per_query=TOP_K_PER_QUERY,
                            top_k_final=TOP_K_FINAL
                        )
                    
                    if not results:
                        st.error("❌ No relevant information found.")
                    else:
                        # Show sources
                        with st.expander(f"📚 Top {len(results)} Sources (reranked)"):
                            for i, result in enumerate(results, 1):
                                st.write(
                                    f"**{i}.** 📄 {result['source']} "
                                    f"(page {result['page']}, {result['type']}) - "
                                    f"Score: {result['score']:.3f}"
                                )
                        
                        # Build context
                        context = "\n\n".join([r["text"] for r in results])
                        
                        # Build prompt with history
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
                            
                            # Show stats
                            stats_msg = f"ℹ️ Used {len(all_queries)} queries ({detected_lang}"
                            if ENABLE_TRANSLATION and len(all_queries) == 6:
                                stats_msg += f" + {'EN' if detected_lang == 'FR' else 'FR'}"
                            stats_msg += f") → {len(results)} unique chunks (reranked)"
                            st.info(stats_msg)
                            
                            # Save to history
                            st.session_state.history.append({
                                "query": query,
                                "answer": answer
                            })
                            
                        except Exception as e:
                            st.error(f"❌ Generation error: {e}")