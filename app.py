"""Streamlit UI for the Intelligent Document Analysis Agent."""

import os
import time
import requests
import streamlit as st

# ─── Configuration ─────────────────────────────────
API_URL = os.getenv("API_URL", "http://localhost:8000")
API_BASE = f"{API_URL}/api/v1"

# ─── Page Config ───────────────────────────────────
st.set_page_config(
    page_title="Intelligent Document Analysis Agent",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Sidebar ───────────────────────────────────────
with st.sidebar:
    st.title("📄 Doc Analysis Agent")
    st.markdown("---")

    # Health check
    st.subheader("System Status")
    try:
        health = requests.get(f"{API_BASE}/health", timeout=5).json()
        status = health.get("status", "unknown")
        if status == "healthy":
            st.success(f"API: {status}")
        else:
            st.warning(f"API: {status}")

        st.metric("Documents Indexed", health.get("document_count", 0))
        st.caption(f"Vector DB: {health.get('vector_db', 'unknown')}")

        with st.expander("Agent Status"):
            for agent, state in health.get("agents", {}).items():
                st.write(f"**{agent}:** {state}")
    except requests.exceptions.ConnectionError:
        st.error("API is not running")
        st.caption(f"Expected at: {API_URL}")
    except Exception as e:
        st.error(f"Health check failed: {e}")

    st.markdown("---")

    # Document list
    st.subheader("Uploaded Documents")
    try:
        docs_resp = requests.get(f"{API_BASE}/documents", timeout=5).json()
        documents = docs_resp.get("documents", [])
        if documents:
            for doc in documents:
                icon = "✅" if doc["status"] == "completed" else "⏳" if doc["status"] == "processing" else "❌"
                st.write(f"{icon} **{doc['filename']}**")
                st.caption(f"ID: {doc['document_id'][:8]}... | Pages: {doc.get('pages', 0)} | Chunks: {doc.get('chunks_created', 0)}")
        else:
            st.info("No documents uploaded yet")
    except Exception:
        st.caption("Could not load document list")

# ─── Main Content ──────────────────────────────────
st.title("Intelligent Document Analysis Agent")
st.markdown(
    "Upload documents and ask questions. The multi-agent system will extract, "
    "index, and analyze your content using RAG with computer vision."
)

# ─── Tabs ──────────────────────────────────────────
tab_upload, tab_query, tab_about = st.tabs(["📤 Upload", "❓ Query", "ℹ️ About"])

# ─── Upload Tab ────────────────────────────────────
with tab_upload:
    st.header("Upload Document")
    st.markdown("Supported formats: **PDF**, **JPG**, **PNG**, **WEBP** (max 50MB)")

    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["pdf", "jpg", "jpeg", "png", "webp"],
        help="Upload a document to extract and index its content",
    )

    if uploaded_file is not None:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.write(f"**File:** {uploaded_file.name}")
            st.write(f"**Size:** {uploaded_file.size / 1024:.1f} KB")
            st.write(f"**Type:** {uploaded_file.type}")

        with col2:
            if st.button("🚀 Upload & Process", type="primary", use_container_width=True):
                with st.spinner("Uploading and processing..."):
                    try:
                        files = {
                            "file": (
                                uploaded_file.name,
                                uploaded_file.getvalue(),
                                uploaded_file.type,
                            )
                        }
                        response = requests.post(
                            f"{API_BASE}/upload", files=files, timeout=30
                        )

                        if response.status_code == 200:
                            data = response.json()
                            st.success(f"Document uploaded successfully!")
                            st.json({
                                "document_id": data["document_id"],
                                "status": data["status"],
                                "filename": data["filename"],
                            })
                            st.info(
                                "Document is being processed in the background. "
                                "Check the sidebar for status updates."
                            )
                        else:
                            st.error(f"Upload failed: {response.text}")

                    except requests.exceptions.ConnectionError:
                        st.error("Could not connect to the API. Is it running?")
                    except Exception as e:
                        st.error(f"Upload error: {e}")

# ─── Query Tab ─────────────────────────────────────
with tab_query:
    st.header("Ask a Question")

    # Document filter
    doc_filter = st.text_input(
        "Document ID (optional)",
        placeholder="Leave empty to search all documents",
        help="Paste a specific document ID to limit the search scope",
    )

    # Query input
    query = st.text_area(
        "Your Question",
        placeholder="e.g., What was the Q3 revenue? / Summarize the methodology section",
        height=100,
    )

    col1, col2 = st.columns([1, 3])
    with col1:
        top_k = st.slider("Results to retrieve", 1, 20, 5)
    with col2:
        pass

    if st.button("🔍 Ask", type="primary", disabled=not query):
        with st.spinner("Analyzing documents..."):
            try:
                payload = {
                    "query": query,
                    "top_k": top_k,
                    "include_images": True,
                }
                if doc_filter.strip():
                    payload["document_id"] = doc_filter.strip()

                response = requests.post(
                    f"{API_BASE}/query", json=payload, timeout=60
                )

                if response.status_code == 200:
                    data = response.json()

                    # Answer
                    st.subheader("Answer")
                    st.markdown(data["answer"])

                    # Metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        confidence = data.get("confidence", 0)
                        color = "green" if confidence > 0.7 else "orange" if confidence > 0.4 else "red"
                        st.metric("Confidence", f"{confidence:.0%}")
                    with col2:
                        st.metric("Processing Time", f"{data.get('processing_time_ms', 0)}ms")
                    with col3:
                        st.metric("Sources Found", len(data.get("sources", [])))

                    # Sources
                    sources = data.get("sources", [])
                    if sources:
                        st.subheader("Sources")
                        for i, src in enumerate(sources, 1):
                            with st.expander(
                                f"Source {i} — Page {src.get('page', '?')} "
                                f"(Relevance: {src.get('relevance_score', 0):.2f})"
                            ):
                                st.write(f"**Document:** {src.get('document_id', '')[:8]}...")
                                st.write(f"**Section:** {src.get('section', 'N/A')}")
                                st.text(src.get("chunk_text", ""))

                elif response.status_code == 404:
                    st.warning("Document not found. Check the document ID.")
                else:
                    st.error(f"Query failed: {response.text}")

            except requests.exceptions.ConnectionError:
                st.error("Could not connect to the API. Is it running?")
            except Exception as e:
                st.error(f"Query error: {e}")

# ─── About Tab ─────────────────────────────────────
with tab_about:
    st.header("About This System")

    st.markdown("""
    ### Architecture
    This is a **Multi-Agent RAG System** that combines:

    - **CrewAI** for multi-agent orchestration
    - **LangChain** for RAG pipeline management
    - **HuggingFace** sentence-transformers for embeddings (all-MiniLM-L6-v2, 384-dim)
    - **Supabase pgvector** for vector storage, semantic search & hybrid search
    - **OpenCV** for computer vision (table/chart detection)
    - **OpenRouter** LLM gateway (Gemini, GPT-4o, Claude, etc.)
    - **FastAPI** for the backend REST API
    - **Supabase Storage** for uploaded file persistence

    ### Agents
    | Agent | Role |
    |-------|------|
    | **Document Processor** | Extracts text, tables, and images from PDFs and scanned docs |
    | **Retrieval Specialist** | Chunks content, generates embeddings, performs semantic search |
    | **Analysis Expert** | Generates answers with citations using multi-modal LLM |

    ### API Endpoints
    | Method | Endpoint | Description |
    |--------|----------|-------------|
    | POST | `/api/v1/upload` | Upload a document |
    | POST | `/api/v1/query` | Ask a question |
    | GET | `/api/v1/documents` | List documents |
    | GET | `/api/v1/documents/{id}` | Get document info |
    | GET | `/api/v1/health` | Health check |

    ### Author
    **M Karthikeya Reddy** — Data Scientist / AI-ML Engineer
    """)
