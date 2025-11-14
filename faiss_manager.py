# faiss_manager.py - Simple FAISS Manager with Git Sync
import os
import faiss
import pickle
import subprocess

DB_DIR = "db"

def load_index():
    """Load FAISS index and metadata if they exist."""
    index_path = os.path.join(DB_DIR, "index.faiss")
    meta_path = os.path.join(DB_DIR, "metadata.pkl")
    
    if os.path.exists(index_path) and os.path.exists(meta_path):
        try:
            if os.path.getsize(index_path) > 0:
                index = faiss.read_index(index_path)
                with open(meta_path, "rb") as f:
                    metadata = pickle.load(f)
                print(f"✅ Index loaded: {len(metadata)} segments")
                return index, metadata
        except Exception as e:
            print(f"⚠️ Error loading index: {e}")
    
    return None, []

def save_index(index, metadata):
    """Save FAISS index and metadata."""
    os.makedirs(DB_DIR, exist_ok=True)
    index_path = os.path.join(DB_DIR, "index.faiss")
    meta_path = os.path.join(DB_DIR, "metadata.pkl")
    
    try:
        faiss.write_index(index, index_path)
        with open(meta_path, "wb") as f:
            pickle.dump(metadata, f)
        print(f"💾 Index saved: {len(metadata)} segments")
        return True
    except Exception as e:
        print(f"❌ Error saving: {e}")
        return False

def clear_index():
    """Delete FAISS index and metadata files."""
    index_path = os.path.join(DB_DIR, "index.faiss")
    meta_path = os.path.join(DB_DIR, "metadata.pkl")
    
    for path in [index_path, meta_path]:
        if os.path.exists(path):
            os.remove(path)
    print("🧹 Index cleared")

def configure_git():
    """Configure Git with credentials from Streamlit secrets."""
    try:
        import streamlit as st
        
        # Set user info
        user = st.secrets.get("GIT_USER_NAME", "streamlit-app")
        email = st.secrets.get("GIT_USER_EMAIL", "app@streamlit.io")
        
        subprocess.run(["git", "config", "user.name", user], check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", email], check=True, capture_output=True)
        
        # Configure authenticated remote if token exists
        token = st.secrets.get("GH_TOKEN", "")
        if token:
            result = subprocess.run(
                ["git", "remote", "get-url", "origin"],
                capture_output=True,
                text=True,
                check=True
            )
            origin_url = result.stdout.strip()
            
            # Convert to authenticated URL
            if "github.com" in origin_url:
                repo_path = origin_url.split("github.com/")[-1].replace(".git", "")
                auth_url = f"https://{token}@github.com/{repo_path}.git"
                subprocess.run(
                    ["git", "remote", "set-url", "origin", auth_url],
                    check=True,
                    capture_output=True
                )
        
        return True
    except Exception as e:
        print(f"Git config error: {e}")
        return False

def push_to_github(commit_message="Update FAISS index"):
    """Push FAISS files and logs to GitHub."""
    try:
        # Configure Git first
        configure_git()
        
        # Add files with force flag to override .gitignore
        subprocess.run(
            ["git", "add", "-f", 
             f"{DB_DIR}/index.faiss", 
             f"{DB_DIR}/metadata.pkl",
             "logs/query_logs.jsonl"],
            check=True,
            capture_output=True
        )
        
        # Commit
        result = subprocess.run(
            ["git", "commit", "-m", commit_message],
            capture_output=True,
            text=True
        )
        
        # Check if there were changes
        if "nothing to commit" in result.stdout:
            return False, "No changes to commit"
        
        # Push
        push_result = subprocess.run(
            ["git", "push"],
            capture_output=True,
            text=True
        )
        
        if push_result.returncode == 0:
            return True, "✅ Pushed to GitHub"
        else:
            return False, f"Push failed: {push_result.stderr}"
    
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.decode() if e.stderr else str(e)
        return False, f"❌ Git error: {error_msg}"
    except Exception as e:
        return False, f"❌ Error: {str(e)}"

def pull_from_github():
    """Pull latest FAISS files from GitHub."""
    try:
        # Configure Git first
        configure_git()
        
        result = subprocess.run(
            ["git", "pull", "--rebase"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            return True, "✅ Pulled from GitHub"
        else:
            return False, f"Pull failed: {result.stderr}"
            
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.decode() if e.stderr else str(e)
        return False, f"❌ Git error: {error_msg}"
    except Exception as e:
        return False, f"❌ Error: {str(e)}"