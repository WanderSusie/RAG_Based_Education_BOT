"""
ui.py
-----
Streamlit front-end for the RAG chatbot.

Run locally with:
    streamlit run ui.py
"""

import os
import uuid

import streamlit as st

from app import RAGPipeline

st.set_page_config(page_title="RAG PDF Chatbot", page_icon="📄", layout="wide")

st.title("📄 RAG PDF Chatbot")
st.caption("Upload a PDF, then ask questions about its content — answered by Groq, grounded in your document.")

# ------------------------------------------------------------------
# Session state setup
# ------------------------------------------------------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]

if "pipeline" not in st.session_state:
    st.session_state.pipeline = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # [{"role": "user"/"assistant", "content": str}]

if "ingested" not in st.session_state:
    st.session_state.ingested = False

if "last_file_name" not in st.session_state:
    st.session_state.last_file_name = None


@st.cache_resource(show_spinner=False)
def get_pipeline(session_id: str) -> RAGPipeline:
    """
    Cached per session_id so the embedding model is loaded only once,
    and each browser session gets its own isolated Chroma collection.
    """
    return RAGPipeline(
        collection_name=f"pdf_docs_{session_id}",
        persist_directory=f"vector_store/{session_id}",
    )


# ------------------------------------------------------------------
# Sidebar: upload + settings
# ------------------------------------------------------------------
with st.sidebar:
    st.header("1. Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    with st.expander("⚙️ Chunking settings"):
        chunk_size = st.slider("Chunk size", 200, 2000, 500, 50)
        chunk_overlap = st.slider("Chunk overlap", 0, 500, 50, 10)

    with st.expander("⚙️ Retrieval settings"):
        top_k = st.slider("Top K chunks", 1, 10, 5)
        score_threshold = st.slider("Similarity threshold", 0.0, 1.0, 0.0, 0.05)

    process_btn = st.button("Process PDF", type="primary", use_container_width=True)

    if process_btn:
        if uploaded_file is None:
            st.warning("Please upload a PDF file first.")
        elif not os.getenv("GROQ_API_KEY"):
            st.error("GROQ_API_KEY is not set. Add it to your .env file and restart the app.")
        else:
            try:
                with st.spinner("Loading models and processing PDF... (first run downloads the embedding model)"):
                    pipeline = get_pipeline(st.session_state.session_id)
                    if st.session_state.ingested:
                        pipeline.vector_store.reset()
                    st.session_state.pipeline = pipeline

                    stats = pipeline.ingest_uploaded_file(
                        uploaded_file, chunk_size=chunk_size, chunk_overlap=chunk_overlap
                    )
                    st.session_state.ingested = True
                    st.session_state.chat_history = []
                    st.session_state.last_file_name = uploaded_file.name
                st.success(f"Processed **{uploaded_file.name}**: {stats['num_pages']} pages → {stats['num_chunks']} chunks.")
            except Exception as e:
                st.error(f"Failed to process PDF: {e}")

    if st.session_state.ingested:
        st.caption(f"📎 Active document: **{st.session_state.last_file_name}**")

    if st.button("Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# ------------------------------------------------------------------
# Main chat area
# ------------------------------------------------------------------
if not st.session_state.ingested:
    st.info("👈 Upload a PDF and click **Process PDF** in the sidebar to get started.")
else:
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    query = st.chat_input("Ask a question about the document...")

    if query:
        st.session_state.chat_history.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving relevant chunks and generating answer..."):
                # last few turns as context for the LLM (excluding the message just added)
                lc_history = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.chat_history[:-1]
                ][-6:]

                try:
                    result = st.session_state.pipeline.query(
                        query,
                        top_k=top_k,
                        score_threshold=score_threshold,
                        chat_history=lc_history,
                    )
                except Exception as e:
                    result = {"answer": f"⚠️ Error while generating answer: {e}", "sources": []}

                st.markdown(result["answer"])

                if result["sources"]:
                    with st.expander(f"📚 Sources ({len(result['sources'])} chunks)"):
                        for src in result["sources"]:
                            meta = src["metadata"]
                            page = meta.get("page_label", meta.get("page", "?"))
                            st.markdown(
                                f"**Rank {src['rank']} · Page {page} · "
                                f"Similarity {src['similarity_score']:.3f}**"
                            )
                            preview = src["document"][:400]
                            st.text(preview + ("..." if len(src["document"]) > 400 else ""))
                            st.divider()

        st.session_state.chat_history.append({"role": "assistant", "content": result["answer"]})
