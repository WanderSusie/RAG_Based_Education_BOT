"""
app.py
------
RAG pipeline used by the Streamlit UI (ui.py).

Stages:
  1. Ingestion    -> load_pdf / split_docs
  2. Embedding    -> EmbeddingManager (sentence-transformers, normalized)
  3. Vector Store -> VectorStoreManager (Chroma, COSINE space, persistent)
  4. Retrieval    -> RAGRetriever  (HYBRID: semantic + BM25 keyword,
                                    fused with Reciprocal Rank Fusion,
                                    + LLM query expansion + cross-encoder rerank)
  5. Generation   -> RAGGenerator (Groq LLM)
"""

import os
import re
import json
import uuid
import tempfile

import numpy as np
from dotenv import load_dotenv
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import chromadb
from groq import Groq

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "is", "are", "was", "were",
    "be", "by", "with", "as", "at", "it", "this", "that", "from", "what", "which", "who",
    "how", "do", "does", "did", "me", "my", "about", "tell", "explain", "please", "can",
}


def tokenize(text: str):
    """Lowercase, split to words, drop stopwords, apply a very light stemmer."""
    tokens = []
    for w in re.findall(r"[a-z0-9]+", text.lower()):
        if w in STOPWORDS:
            continue
        for suf in ("ing", "ed", "es", "s"):
            if len(w) > len(suf) + 3 and w.endswith(suf):
                w = w[: -len(suf)]
                break
        tokens.append(w)
    return tokens


# ============================================================
# 1. Ingestion
# ============================================================
def load_pdf(pdf_path: str):
    return PyPDFLoader(pdf_path).load()


def split_docs(documents, chunk_size: int = 800, chunk_overlap: int = 150):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


# ============================================================
# 2. Embedding
# ============================================================
class EmbeddingManager:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def generate_embeddings(self, texts):
        # normalized -> dot product == cosine similarity
        return self.model.encode(texts, show_progress_bar=False, normalize_embeddings=True)


# ============================================================
# 3. Vector Store
# ============================================================
class VectorStoreManager:
    def __init__(self, persist_directory: str = "vector_store", collection_name: str = "pdf_documents"):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        os.makedirs(persist_directory, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self._get_collection()

    def _get_collection(self):
        # hnsw:space=cosine fixes the L2-vs-cosine bug (similarity = 1 - distance is now valid)
        return self.client.get_or_create_collection(
            name=self.collection_name, metadata={"hnsw:space": "cosine"}
        )

    def add_documents(self, documents, embeddings):
        if len(documents) != len(embeddings):
            raise ValueError("num of documents does not match num of embeddings")
        ids, metas, texts, embs = [], [], [], []
        for i, (doc, emb) in enumerate(zip(documents, embeddings)):
            ids.append(f"doc_{uuid.uuid4()}")
            m = dict(doc.metadata)
            m["doc_index"] = i
            m["content_length"] = len(doc.page_content)
            metas.append(m)
            texts.append(doc.page_content)
            embs.append(emb.tolist())
        self.collection.add(ids=ids, metadatas=metas, documents=texts, embeddings=embs)
        return len(texts)

    def reset(self):
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.collection = self._get_collection()


# ============================================================
# 4. Retrieval  (hybrid search)
# ============================================================
class RAGRetriever:
    """
    Hybrid retriever:
      - semantic search  (cosine over sentence embeddings)
      - keyword search   (BM25) -> catches exact names / codes / rare terms
      - Reciprocal Rank Fusion to merge both, across multiple query variants
      - optional cross-encoder reranking for precision
    """

    def __init__(self, embedding_manager: EmbeddingManager, vector_store: VectorStoreManager):
        self.embedding_manager = embedding_manager
        self.vector_store = vector_store
        self.reranker = None
        self.ids, self.texts, self.metas = [], [], []
        self.emb = np.zeros((0, 1), dtype=np.float32)
        self.bm25 = None

    def build_index(self):
        """Load the persisted chunks into memory and build the BM25 index."""
        data = self.vector_store.collection.get(include=["documents", "metadatas", "embeddings"])
        self.ids = list(data["ids"])
        self.texts = list(data["documents"])
        self.metas = list(data["metadatas"])
        if len(self.ids) == 0:
            self.emb, self.bm25 = np.zeros((0, 1), dtype=np.float32), None
            return
        self.emb = np.array(data["embeddings"], dtype=np.float32)
        self.bm25 = BM25Okapi([tokenize(t) or ["_"] for t in self.texts])

    def _get_reranker(self):
        if self.reranker is None:
            self.reranker = CrossEncoder(RERANK_MODEL)
        return self.reranker

    def retrieve(self, query, top_k=5, score_threshold=0.0, mode="hybrid", use_rerank=True, extra_queries=None):
        if not self.ids:
            return []
        queries = [query] + [q for q in (extra_queries or []) if q and q != query]
        qvecs = self.embedding_manager.generate_embeddings(queries)
        cand_k = min(len(self.ids), max(top_k * 4, 20))
        w_dense = 0.0 if mode == "keyword" else 1.0
        w_bm25 = 0.0 if mode == "semantic" else 1.0
        K = 60  # RRF constant

        fused, best_cos, best_bm25 = {}, {}, {}
        for qi, (q, qv) in enumerate(zip(queries, qvecs)):
            qw = 1.0 if qi == 0 else 0.7
            sims = self.emb @ qv
            for r, idx in enumerate(np.argsort(-sims)[:cand_k]):
                idx = int(idx)
                best_cos[idx] = max(best_cos.get(idx, -1), float(sims[idx]))
                fused[idx] = fused.get(idx, 0) + qw * w_dense / (K + r + 1)
            if self.bm25 is not None and w_bm25 > 0:
                toks = tokenize(q)
                if toks:
                    bs = self.bm25.get_scores(toks)
                    for r, idx in enumerate(np.argsort(-bs)[:cand_k]):
                        idx = int(idx)
                        if bs[idx] <= 0:
                            continue
                        best_bm25[idx] = max(best_bm25.get(idx, 0), float(bs[idx]))
                        fused[idx] = fused.get(idx, 0) + qw * w_bm25 / (K + r + 1)

        # cosine for every candidate (even keyword-only hits)
        for idx in fused:
            if idx not in best_cos:
                best_cos[idx] = float(max(self.emb[idx] @ qv for qv in qvecs))

        ranked = sorted(fused, key=fused.get, reverse=True)
        ranked = [i for i in ranked if best_cos[i] >= score_threshold or best_bm25.get(i, 0) > 0]
        pool = ranked[: max(top_k * 3, 15)] if use_rerank else ranked[:top_k]

        rerank_scores = {}
        if use_rerank and len(pool) > 1:
            try:
                scores = self._get_reranker().predict([(query, self.texts[i]) for i in pool])
                rerank_scores = {i: float(s) for i, s in zip(pool, scores)}
                pool = sorted(pool, key=lambda i: rerank_scores[i], reverse=True)
            except Exception:
                pass

        results = []
        for rank, i in enumerate(pool[:top_k], start=1):
            results.append({
                "id": self.ids[i],
                "document": self.texts[i],
                "metadata": self.metas[i],
                "similarity_score": best_cos[i],
                "distance": 1 - best_cos[i],
                "bm25_score": best_bm25.get(i, 0.0),
                "rerank_score": rerank_scores.get(i),
                "rank": rank,
            })
        return results


# ============================================================
# 5. Generation
# ============================================================
class RAGGenerator:
    def __init__(self, api_key: str = GROQ_API_KEY, model: str = GROQ_MODEL):
        if not api_key:
            raise ValueError("GROQ_API_KEY not found. Add it to your .env file or environment.")
        self.client = Groq(api_key=api_key)
        self.model = model

    def _extra(self):
        return {"reasoning_effort": "low"} if "gpt-oss" in self.model else {}

    @staticmethod
    def build_context(retrieved_docs) -> str:
        parts = []
        for d in retrieved_docs:
            page = d["metadata"].get("page_label", d["metadata"].get("page", "?"))
            parts.append(f"[Page {page}]\n{d['document']}")
        return "\n\n---\n\n".join(parts)

    def expand_query(self, question: str, chat_history=None) -> dict:
        """Rewrite follow-ups into a standalone query and add alternative phrasings/keywords."""
        history = ""
        for m in (chat_history or [])[-4:]:
            history += f"{m['role']}: {m['content'][:300]}\n"
        prompt = (
            "You help a document search engine. Given the chat history and the latest question, "
            "return ONLY a JSON object: "
            '{"standalone": "<question rewritten to be self-contained>", '
            '"alternatives": ["<paraphrase using synonyms>", "<keywords only>", "<related technical terms>"]}\n\n'
            f"Chat history:\n{history or '(none)'}\nLatest question: {question}"
        )
        try:
            r = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2, max_tokens=800, **self._extra(),
            )
            m = re.search(r"\{.*\}", r.choices[0].message.content or "", re.S)
            data = json.loads(m.group(0))
            return {
                "standalone": data.get("standalone") or question,
                "alternatives": [str(a) for a in data.get("alternatives", [])][:3],
            }
        except Exception:
            return {"standalone": question, "alternatives": []}

    def generate(self, query: str, retrieved_docs, chat_history=None) -> str:
        system_prompt = (
            "You are an expert assistant answering questions about the user's uploaded PDF. "
            "Use the context excerpts below as your source of truth. The wording of the question may differ "
            "from the document (synonyms, paraphrases, abbreviations) — match by meaning, not exact words. "
            "Give a clear, well-structured answer (use bullets when helpful) and mention page numbers. "
            "If the context only partly answers the question, answer what you can and say what is missing. "
            "Only if the context is truly unrelated, say the document doesn't contain that information. "
            "Never invent facts that are not in the context.\n\n"
            f"Context:\n{self.build_context(retrieved_docs)}"
        )
        messages = [{"role": "system", "content": system_prompt}]
        if chat_history:
            messages.extend(chat_history)
        messages.append({"role": "user", "content": query})
        r = self.client.chat.completions.create(
            model=self.model, messages=messages, temperature=0.2, max_tokens=2048, **self._extra()
        )
        return r.choices[0].message.content


# ============================================================
# End-to-end pipeline
# ============================================================
class RAGPipeline:
    def __init__(self, collection_name: str = "pdf_documents", persist_directory: str = "vector_store"):
        self.embedding_manager = EmbeddingManager()
        self.vector_store = VectorStoreManager(persist_directory, collection_name)
        self.retriever = RAGRetriever(self.embedding_manager, self.vector_store)
        self.generator = RAGGenerator()
        self.retriever.build_index()

    def ingest_pdf(self, pdf_path: str, chunk_size: int = 800, chunk_overlap: int = 150, source_name=None) -> dict:
        documents = load_pdf(pdf_path)
        if source_name:
            for d in documents:
                d.metadata["source"] = source_name
        chunks = split_docs(documents, chunk_size, chunk_overlap)
        chunks = [c for c in chunks if c.page_content.strip()]
        if not chunks:
            raise ValueError("No extractable text found (is this a scanned/image-only PDF?).")
        embeddings = self.embedding_manager.generate_embeddings([c.page_content for c in chunks])
        added = self.vector_store.add_documents(chunks, embeddings)
        self.retriever.build_index()
        words = sum(len(c.page_content.split()) for c in chunks)
        return {"num_pages": len(documents), "num_chunks": len(chunks), "num_added": added, "num_words": words}

    def ingest_uploaded_file(self, uploaded_file, chunk_size: int = 800, chunk_overlap: int = 150) -> dict:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name
        try:
            return self.ingest_pdf(tmp_path, chunk_size, chunk_overlap, source_name=uploaded_file.name)
        finally:
            os.remove(tmp_path)

    def query(self, question: str, top_k: int = 5, score_threshold: float = 0.0, chat_history=None,
              mode: str = "hybrid", use_expansion: bool = True, use_rerank: bool = True) -> dict:
        search_q, extra = question, []
        if use_expansion:
            exp = self.generator.expand_query(question, chat_history)
            search_q, extra = exp["standalone"], exp["alternatives"]

        docs = self.retriever.retrieve(search_q, top_k=top_k, score_threshold=score_threshold,
                                       mode=mode, use_rerank=use_rerank, extra_queries=extra)
        queries = [search_q] + extra
        if not docs:
            return {"answer": "I couldn't find relevant content in the document for this question. "
                              "Try rephrasing, or lower the similarity threshold in the sidebar.",
                    "sources": [], "queries": queries}
        return {"answer": self.generator.generate(question, docs, chat_history),
                "sources": docs, "queries": queries}


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print('Usage: python app.py <pdf_path> "<question>"')
        raise SystemExit(1)
    p = RAGPipeline()
    p.vector_store.reset()
    print("Ingested:", p.ingest_pdf(sys.argv[1]))
    res = p.query(sys.argv[2])
    print("\nSearch queries:", res["queries"])
    print("\nAnswer:\n", res["answer"])
