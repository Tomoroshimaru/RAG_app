# 🧠 RAG PDF App - Cross-Lingual Retrieval Augmented Generation

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=for-the-badge&logo=openai&logoColor=white)](https://openai.com/)

A powerful **Retrieval-Augmented Generation (RAG)** application that enables intelligent document querying with **cross-lingual search** (FR+EN), **GitHub synchronization**, and **real-time analytics**.

--- 

## 🚀 Features

### 🔍 **Core RAG Capabilities**
- 📂 **PDF Upload & Processing**: Extract text and tables from multiple PDFs
- 🧠 **Semantic Embeddings**: Multilingual embeddings with `intfloat/multilingual-e5-base`
- 🗃️ **Vector Search**: FAISS-based similarity search with cosine distance
- 💬 **Answer Generation**: Context-aware responses using GPT-4o-mini

### 🌍 **Cross-Lingual Search**
- 🔍 **Auto Language Detection**: Detects FR/EN automatically with `langdetect`
- 🔄 **Multi-Query Rewriting**: Generates 2 reformulations in source language
- 🌐 **Translation**: Automatically translates queries to other language
- 📊 **6 Query Variants**: 3 in FR + 3 in EN for maximum recall
- 🎯 **Smart Reranking**: Deduplicates and ranks by max similarity score

### 🔗 **GitHub Synchronization**
- ⬆️ **Push to GitHub**: Sync FAISS index, metadata, and logs
- ⬇️ **Pull from GitHub**: Load latest data across deployments
- 🔒 **Authenticated Operations**: Secure Git operations with PAT

### 📊 **Analytics Dashboard**
- 🌐 **Language Breakdown**: Visual distribution of FR/EN queries
- 📄 **Document Stats**: Most referenced documents
- 📈 **Similarity Scores**: Average search relevance
- 📘 **Query Logs**: Full-text searchable history

---

## 📁 Project Structure

```
RAG_app/
├── rag_app.py              # Main Streamlit application
├── faiss_manager.py        # FAISS index + Git sync utilities
├── log_manager.py          # Logging system (JSONL format)
│
├── db/
│   ├── .gitkeep           # Tracked in Git
│   ├── index.faiss        # FAISS vector index (synced)
│   └── metadata.pkl       # Document metadata (synced)
│
├── logs/
│   ├── .gitkeep           # Tracked in Git
│   └── query_logs.jsonl   # Query logs (synced)
│
├── data/
│   └── .gitkeep           # PDFs stored here (not synced)
│
├── requirements.txt        # Python dependencies
├── .gitignore             # Git ignore rules
└── README.md              # This file
```

---

## ⚙️ Installation

### **1. Clone the Repository**
```bash
git clone https://github.com/yourusername/RAG_app.git
cd RAG_app

chmod +x setup.sh
source setup.sh
```

**To create a GitHub Personal Access Token:**
1. GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Generate new token with `repo` scope
3. Copy the token immediately

---

## 🚀 Usage

### **Local Development**
```bash
streamlit run rag_app.py
```

### **Deploy to Streamlit Cloud**
1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io/)
3. Deploy from your repository
4. Set main file: `rag_app.py`
5. Add secrets in **App settings → Secrets**

---

## 📖 How to Use

### **1. Upload Documents**
1. Go to **"📤 Upload Documents"** tab
2. Drag & drop PDF files
3. Click **"🚀 Index Documents"**
4. Wait for processing and embedding creation

### **2. Configure Search**
In the sidebar:
- ✅ **Enable Query Rewriting**: Generates 3 query variants
- ✅ **Enable Cross-Lingual Search**: Searches in FR + EN (6 queries total)

### **3. Ask Questions**
1. Go to **"❓ Ask Questions"** tab
2. Type your question in French or English
3. Click **"🔍 Search"**
4. View:
   - Detected language
   - Query variants generated
   - Translations (if enabled)
   - Sources used
   - Generated answer

### **4. View Logs**
1. Go to **"📘 Logs"** tab
2. Browse all query and upload logs
3. Use search to filter logs
4. View detailed JSON data

### **5. Analytics Dashboard**
1. Click **"📊 Show Analytics Dashboard"** in sidebar
2. View:
   - Language distribution
   - Most referenced documents
   - Average similarity scores
   - Upload statistics

### **6. Sync with GitHub**
- **⬇️ Pull from GitHub**: Load latest data
- **⬆️ Push to GitHub**: Save FAISS index + metadata + logs
- **🗑️ Clear Database**: Delete all indexed documents
- **🧹 Clear Conversation**: Reset chat history
- **🗑️ Clear Logs**: Delete all logs

---

## 🔧 Configuration

### **Model Settings**
```python
MODEL_NAME = "intfloat/multilingual-e5-base"  # Embedding model
CHUNK_SIZE = 500                               # Text chunk size
TOP_K_PER_QUERY = 5                           # Results per query variant
TOP_K_FINAL = 15                              # Final results after reranking
```

### **Search Modes**

| Mode | Queries | Chunks | Use Case |
|------|---------|--------|----------|
| **Simple** | 1 | ~5 | Fast, single-language docs |
| **Multi-Query** | 3 | ~12 | Better recall, monolingual |
| **Cross-Lingual** | 6 | ~15 | Max recall, multilingual docs |

---

## 🌍 Cross-Lingual Search Workflow

```
User Query (FR or EN)
    ↓
[Detect Language] → FR or EN
    ↓
[Rewrite × 2] → 3 queries in source language
    ↓
[Translate × 3] → 3 queries in other language
    ↓
[6 Queries Total]
    ↓
[FAISS Search: 5 chunks × 6 queries = 30 chunks]
    ↓
[Deduplication + Reranking by max score]
    ↓
[Keep Top 15 Chunks]
    ↓
[GPT-4o-mini generates answer from context]
```

---

## 📊 Log Format

### **Query Log**
```json
{
  "timestamp": "2024-11-14T15:30:45.123456",
  "type": "query",
  "data": {
    "query": "What are the benefits of AI?",
    "detected_language": "EN",
    "query_variants": [
      "What are the benefits of AI?",
      "What are the advantages of artificial intelligence?",
      "Quels sont les avantages de l'IA?",
      "..."
    ],
    "results": [
      {
        "text": "...",
        "source": "document.pdf",
        "page": 5,
        "type": "text",
        "score": 0.89
      }
    ],
    "answer": "The benefits of AI include..."
  }
}
```

### **Upload Log**
```json
{
  "timestamp": "2024-11-14T14:20:30.123456",
  "type": "upload",
  "data": {
    "filenames": ["doc1.pdf", "doc2.pdf"],
    "segments_added": 150,
    "total_segments": 450
  }
}
```

---

## 📈 Performance

### **Response Times**
| Mode | Average Time |
|------|--------------|
| Simple (1 query) | ~2-3s |
| Multi-query (3 queries) | ~3-4s |
| Cross-lingual (6 queries) | ~5-6s |

### **API Costs** (GPT-4o-mini)
| Mode | API Calls | Tokens | Cost/query |
|------|-----------|--------|------------|
| Simple | 1 (answer) | ~1000 | ~$0.0015 |
| Multi-query | 2 (rewrite + answer) | ~1300 | ~$0.002 |
| Cross-lingual | 3 (rewrite + translate + answer) | ~1600 | ~$0.0024 |

---

## 🛠️ Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io/) 1.33.0
- **Embeddings**: [Sentence Transformers](https://www.sbert.net/) - `intfloat/multilingual-e5-base`
- **Vector DB**: [FAISS](https://github.com/facebookresearch/faiss) (CPU)
- **LLM**: [OpenAI GPT-4o-mini](https://openai.com/)
- **PDF Processing**: [pdfplumber](https://github.com/jsvine/pdfplumber)
- **Language Detection**: [langdetect](https://pypi.org/project/langdetect/)
- **Version Control**: Git + GitHub

---

## 🐛 Troubleshooting

### **Issue: "Permission denied" on push**
**Solution**: Check that `GH_TOKEN` has `repo` scope. Regenerate if needed.

### **Issue: "db is ignored by .gitignore"**
**Solution**: Use the provided `.gitignore` which allows `db/` and `logs/` folders.

### **Issue: Logs not appearing**
**Solution**: Logs are created after first query. Check `logs/query_logs.jsonl` exists.

### **Issue: Rewriting/Translation fails**
**Solution**: Check `OPENAI_API_KEY` is valid. App will fallback to original query.

### **Issue: Language detection fails**
**Solution**: Install `langdetect` with `pip install langdetect`. Fallback is EN.

---

## 📝 Dependencies

```txt
streamlit==1.33.0
pdfplumber==0.10.3
pdfminer.six==20221105
sentence-transformers
faiss-cpu
numpy
openai
langdetect
pandas
```

---

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [Sentence Transformers](https://www.sbert.net/) for multilingual embeddings
- [FAISS](https://github.com/facebookresearch/faiss) for efficient vector search
- [OpenAI](https://openai.com/) for GPT-4o-mini
- [Streamlit](https://streamlit.io/) for the amazing framework

---

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

**Built with ❤️ for better document understanding**