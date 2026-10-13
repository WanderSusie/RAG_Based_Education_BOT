"""
ui.py
-----
Streamlit front-end for the RAG chatbot (modern styled UI).

Run locally with:
    streamlit run ui.py
"""

import os
import re
import html
import uuid

import streamlit as st

from app import RAGPipeline, tokenize

st.set_page_config(page_title="DocMind · RAG PDF Chatbot", page_icon="🧠", layout="wide")

# ------------------------------------------------------------------
# Styling
# ------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.block-container { padding-top: 1.5rem; max-width: 1100px; }

.hero {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #db2777 100%);
    border-radius: 20px; padding: 28px 34px; color: white; margin-bottom: 1.2rem;
    box-shadow: 0 10px 30px rgba(79,70,229,.30);
}
.hero h1 { margin: 0; font-size: 2.1rem; font-weight: 700; color: white; }
.hero p  { margin: .4rem 0 0; opacity: .92; font-size: 1.02rem; }

.stat-row { display: flex; gap: 12px; margin-bottom: 1rem; flex-wrap: wrap; }
.stat {
    flex: 1; min-width: 120px; background: rgba(124,58,237,.08);
    border: 1px solid rgba(124,58,237,.25); border-radius: 14px; padding: 12px 16px; text-align: center;
}
.stat b { display: block; font-size: 1.5rem; color: #7c3aed; }
.stat span { font-size: .78rem; opacity: .75; text-transform: uppercase; letter-spacing: .05em; }

.step {
    background: rgba(124,58,237,.07); border: 1px solid rgba(124,58,237,.2);
    border-radius: 16px; padding: 20px; height: 100%;
}
.step .num {
    display: inline-block; width: 32px; height: 32px; line-height: 32px; text-align: center;
    border-radius: 50%; background: linear-gradient(135deg,#4f46e5,#db2777); color: white; font-weight: 700; margin-bottom: 8px;
}

.src-card {
    border-left: 4px solid #7c3aed; background: rgba(124,58,237,.06);
    border-radius: 0 12px 12px 0; padding: 10px 14px; margin-bottom: 10px; font-size: .9rem;
}
.badge {
    display: inline-block; padding: 2px 9px; border-radius: 999px; font-size: .72rem;
    font-weight: 600; margin-right: 6px; background: rgba(124,58,237,.15); color: #7c3aed;
}
.badge.kw { background: rgba(219,39,119,.15); color: #db2777; }
.src-card mark { background: #fde68a; color: #111; padding: 0 2px; border-radius: 3px; }
.qchip {
    display: inline-block; background: rgba(79,70,229,.12); color: #4f46e5; border-radius: 999px;
    padding: 3px 12px; margin: 2px 4px 2px 0; font-size: .8rem;
}
section[data-testid="stSidebar"] { border-right: 1px solid rgba(124,58,237,.2); }
div.stButton > button[kind="primary"] {
    background: linear-gradient(135deg,#4f46e5,#7c3aed); border: none; font-weight: 600;
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="hero">
  <h1>🧠 DocMind</h1>
  <p>Chat with any PDF — hybrid semantic + keyword search, grounded answers with page citations.</p>
</div>
""",
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------
# Session state
# ------------------------------------------------------------------
defaults = {
    "session_id": str(uuid.uuid4())[:8],
    "pipeline": None,
    "chat_history": [],   # [{"role", "content", "sources"?, "queries"?}]
    "ingested": False,
    "last_file_name": None,
    "stats": None,
    "pending_question": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


@st.cache_resource(show_spinner=False)
def get_pipeline(session_id: str) -> RAGPipeline:
    return RAGPipeline(
        collection_name=f"pdf_docs_{session_id}",
        persist_directory=f"vector_store/{session_id}",
    )


def highlight(text: str, query_terms) -> str:
    """HTML-escape the text and <mark> the words that match the query."""
    safe = html.escape(text)
    words = sorted({w for w in re.findall(r"[A-Za-z0-9]+", " ".join(query_terms)) if len(w) > 2}, key=len, reverse=True)
    stop = {"the", "and", "what", "how", "about", "this", "that", "with", "from", "for"}
    words = [w for w in words if w.lower() not in stop]
    if words:
        pattern = re.compile(r"(" + "|".join(re.escape(w) for w in words) + r")", re.I)
        safe = pattern.sub(r"<mark>\1</mark>", safe)
    return safe


def render_sources(sources, queries):
    if queries:
        chips = "".join(f'<span class="qchip">🔎 {html.escape(q)}</span>' for q in queries)
        st.markdown(f"**Search queries used**<br>{chips}", unsafe_allow_html=True)
    for s in sources:
        meta = s["metadata"]
        page = meta.get("page_label", meta.get("page", "?"))
        badges = f'<span class="badge">Page {page}</span><span class="badge">Semantic {s["similarity_score"]:.2f}</span>'
        if s.get("bm25_score", 0) > 0:
            badges += '<span class="badge kw">Keyword match</span>'
        if s.get("rerank_score") is not None:
            badges += f'<span class="badge">Rerank {s["rerank_score"]:.1f}</span>'
        preview = s["document"][:600] + ("…" if len(s["document"]) > 600 else "")
        st.markdown(
            f'<div class="src-card"><b>#{s["rank"]}</b> {badges}<br><br>'
            f'{highlight(preview, queries)}</div>',
            unsafe_allow_html=True,
        )


# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 📂 Document")
    uploaded_file = st.file_uploader("Drop a PDF here", type=["pdf"], label_visibility="collapsed")

    with st.expander("✂️ Chunking", expanded=False):
        chunk_size = st.slider("Chunk size", 300, 2000, 800, 50)
        chunk_overlap = st.slider("Chunk overlap", 0, 500, 150, 10)

    process_btn = st.button("⚡ Process PDF", type="primary", use_container_width=True)

    st.markdown("### 🔍 Search")
    mode_label = st.radio(
        "Search mode",
        ["Hybrid (recommended)", "Semantic only", "Keyword only"],
        help="Hybrid combines meaning-based search with BM25 keyword search.",
    )
    mode = {"Hybrid (recommended)": "hybrid", "Semantic only": "semantic", "Keyword only": "keyword"}[mode_label]
    use_expansion = st.toggle("Smart query expansion", value=True,
                              help="LLM rewrites your question and adds synonyms/keywords before searching.")
    use_rerank = st.toggle("Rerank results", value=True,
                           help="Cross-encoder reorders candidates for better precision (slightly slower).")
    top_k = st.slider("Chunks to use (Top K)", 1, 15, 6)
    score_threshold = st.slider("Min. semantic similarity", 0.0, 1.0, 0.0, 0.05,
                                help="Keep at 0 for best recall. Keyword hits always pass.")

    if process_btn:
        if uploaded_file is None:
            st.warning("Please upload a PDF first.")
        elif not os.getenv("GROQ_API_KEY"):
            st.error("GROQ_API_KEY is not set. Add it to your .env file and restart.")
        else:
            try:
                with st.spinner("Reading, chunking and indexing your PDF…"):
                    pipeline = get_pipeline(st.session_state.session_id)
                    if st.session_state.ingested:
                        pipeline.vector_store.reset()
                    st.session_state.pipeline = pipeline
                    stats = pipeline.ingest_uploaded_file(uploaded_file, chunk_size, chunk_overlap)
                    st.session_state.ingested = True
                    st.session_state.chat_history = []
                    st.session_state.last_file_name = uploaded_file.name
                    st.session_state.stats = stats
                st.success("Document ready!")
            except Exception as e:
                st.error(f"Failed to process PDF: {e}")

    st.divider()
    c1, c2 = st.columns(2)
    if c1.button("🗑️ Clear", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()
    if st.session_state.chat_history:
        transcript = "\n\n".join(f"{m['role'].upper()}: {m['content']}" for m in st.session_state.chat_history)
        c2.download_button("💾 Export", transcript, "chat.txt", use_container_width=True)

# ------------------------------------------------------------------
# Main area
# ------------------------------------------------------------------
if not st.session_state.ingested:
    cols = st.columns(3)
    steps = [
        ("1", "Upload", "Drop a PDF in the sidebar — notes, papers, textbooks, reports."),
        ("2", "Process", "Click <b>Process PDF</b>. It is chunked and indexed for semantic + keyword search."),
        ("3", "Ask", "Ask in your own words. You don't need the exact words used in the document."),
    ]
    for col, (n, t, d) in zip(cols, steps):
        col.markdown(f'<div class="step"><div class="num">{n}</div><h4>{t}</h4><p>{d}</p></div>',
                     unsafe_allow_html=True)
else:
    s = st.session_state.stats
    st.markdown(
        f"""<div class="stat-row">
        <div class="stat"><b>📎</b><span>{html.escape(st.session_state.last_file_name)}</span></div>
        <div class="stat"><b>{s['num_pages']}</b><span>Pages</span></div>
        <div class="stat"><b>{s['num_chunks']}</b><span>Chunks</span></div>
        <div class="stat"><b>{s['num_words']:,}</b><span>Words</span></div>
        </div>""",
        unsafe_allow_html=True,
    )

    if not st.session_state.chat_history:
        st.markdown("**💡 Try one of these**")
        suggestions = ["Summarize this document", "What are the key points?", "List the main topics covered"]
        scols = st.columns(len(suggestions))
        for col, q in zip(scols, suggestions):
            if col.button(q, use_container_width=True):
                st.session_state.pending_question = q
                st.rerun()

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"], avatar="🧑‍🎓" if msg["role"] == "user" else "🧠"):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander(f"📚 Sources ({len(msg['sources'])})"):
                    render_sources(msg["sources"], msg.get("queries", []))

    query = st.chat_input("Ask anything about your document…")
    if st.session_state.pending_question:
        query, st.session_state.pending_question = st.session_state.pending_question, None

    if query:
        history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.chat_history][-6:]
        st.session_state.chat_history.append({"role": "user", "content": query})
        with st.chat_message("user", avatar="🧑‍🎓"):
            st.markdown(query)

        with st.chat_message("assistant", avatar="🧠"):
            with st.spinner("Searching the document and thinking…"):
                try:
                    result = st.session_state.pipeline.query(
                        query, top_k=top_k, score_threshold=score_threshold, chat_history=history,
                        mode=mode, use_expansion=use_expansion, use_rerank=use_rerank,
                    )
                except Exception as e:
                    result = {"answer": f"⚠️ Error while generating answer: {e}", "sources": [], "queries": []}

            st.markdown(result["answer"])
            if result["sources"]:
                with st.expander(f"📚 Sources ({len(result['sources'])})"):
                    render_sources(result["sources"], result["queries"])

        st.session_state.chat_history.append({
            "role": "assistant", "content": result["answer"],
            "sources": result["sources"], "queries": result["queries"],
        })
