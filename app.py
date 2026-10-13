"""
app.py
------
End-to-end RAG pipeline used by the Streamlit UI (ui.py).

Pipeline stages (matching the structure of RAG_pipeline.ipynb):
  1. Ingestion   -> load_pdf / split_docs
  2. Embedding   -> EmbeddingManager (sentence-transformers)
  3. Vector Store-> VectorStoreManager (Chroma, persistent)
  4. Retrieval   -> RAGRetriever
  5. Generation  -> RAGGenerator (Groq LLM)

RAGPipeline ties all five stages together and is the only class ui.py
needs to import.
"""

import os
import uuid
import tempfile

from dotenv import load_dotenv
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
from groq import Groq

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# openai/gpt-oss-120b is Groq's current recommended general-purpose model
# (llama-3.3-70b-versatile was deprecated on Groq in 2026).
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")


# ============================================================
# 1. Ingestion
# ============================================================
def load_pdf(pdf_path: str):
    """Load a single PDF file into LangChain Document objects (one per page)."""
    loader = PyPDFLoader(pdf_path)
    return loader.load()


def split_docs(documents, chunk_size: int = 500, chunk_overlap: int = 50):
    """Split page-level documents into smaller overlapping chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return text_splitter.split_documents(documents)


# ============================================================
# 2. Embedding
# ============================================================
class EmbeddingManager:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(self.model_name)

    def generate_embeddings(self, texts):
        return self.model.encode(texts, show_progress_bar=False)


# ============================================================
# 3. Vector Store
# ============================================================
class VectorStoreManager:
    def __init__(self, persist_directory: str = "vector_store", collection_name: str = "pdf_documents"):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        self._initialize_store()

    def _initialize_store(self):
        os.makedirs(self.persist_directory, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "vector store collection for pdf embeddings in RAG"},
        )

    def add_documents(self, documents, embeddings):
        if len(documents) != len(embeddings):
            raise ValueError("num of documents does not match num of embeddings")

        ids, all_metadata, documents_content, embeddings_list = [], [], [], []
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            ids.append(f"doc_{uuid.uuid4()}")
            metadata = dict(doc.metadata)
            metadata["doc_index"] = i
            metadata["content_length"] = len(doc.page_content)
            all_metadata.append(metadata)
            documents_content.append(doc.page_content)
            embeddings_list.append(embedding.tolist())

        self.collection.add(
            ids=ids,
            metadatas=all_metadata,
            documents=documents_content,
            embeddings=embeddings_list,
        )
        return len(documents_content)

    def reset(self):
        """Wipe the collection (used when a new PDF is uploaded in the same session)."""
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "vector store collection for pdf embeddings in RAG"},
        )


# ============================================================
# 4. Retrieval
# ============================================================
class RAGRetriever:
    def __init__(self, embedding_manager: EmbeddingManager, vector_store: VectorStoreManager):
        self.embedding_manager = embedding_manager
        self.vector_store = vector_store

    def retrieve(self, query: str, top_k: int = 5, score_threshold: float = 0.0):
        query_embedding = self.embedding_manager.generate_embeddings([query])[0]

        results = self.vector_store.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
        )

        retrieved_docs = []
        if results["documents"] and results["documents"][0]:
            ids = results["ids"][0]
            metadatas = results["metadatas"][0]
            documents = results["documents"][0]
            distances = results["distances"][0]

            for i, (doc_id, metadata, document, distance) in enumerate(
                zip(ids, metadatas, documents, distances)
            ):
                similarity_score = 1 - distance
                if similarity_score >= score_threshold:
                    retrieved_docs.append(
                        {
                            "id": doc_id,
                            "document": document,
                            "metadata": metadata,
                            "distance": distance,
                            "similarity_score": similarity_score,
                            "rank": i + 1,
                        }
                    )
        return retrieved_docs


# ============================================================
# 5. Generation
# ============================================================
class RAGGenerator:
    def __init__(self, api_key: str = GROQ_API_KEY, model: str = GROQ_MODEL):
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY not found. Add it to your .env file "
                "(see .env.example) or export it as an environment variable."
            )
        self.client = Groq(api_key=api_key)
        self.model = model

    @staticmethod
    def build_context(retrieved_docs) -> str:
        parts = []
        for d in retrieved_docs:
            source = d["metadata"].get("source", "unknown")
            page = d["metadata"].get("page_label", d["metadata"].get("page", "?"))
            parts.append(f"[Source: {source} | Page: {page}]\n{d['document']}")
        return "\n\n---\n\n".join(parts)

    def generate(self, query: str, retrieved_docs, chat_history=None) -> str:
        context = self.build_context(retrieved_docs)

        system_prompt = (
            "You are a helpful assistant that answers questions using ONLY the "
            "context extracted from the user's uploaded PDF, provided below. "
            "If the answer is not contained in the context, say clearly that "
            "the document does not contain that information — do not make "
            "anything up. When useful, mention which page the information "
            "came from.\n\n"
            f"Context:\n{context}"
        )

        messages = [{"role": "system", "content": system_prompt}]
        if chat_history:
            messages.extend(chat_history)
        messages.append({"role": "user", "content": query})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.2,
            max_tokens=1024,
        )
        return response.choices[0].message.content


# ============================================================
# End-to-end pipeline
# ============================================================
class RAGPipeline:
    """Glues ingestion -> embedding -> vector store -> retrieval -> generation."""

    def __init__(self, collection_name: str = "pdf_documents", persist_directory: str = "vector_store"):
        self.embedding_manager = EmbeddingManager()
        self.vector_store = VectorStoreManager(
            persist_directory=persist_directory,
            collection_name=collection_name,
        )
        self.retriever = RAGRetriever(self.embedding_manager, self.vector_store)
        self.generator = RAGGenerator()

    # ---- Ingestion helpers -------------------------------------------------
    def ingest_pdf(self, pdf_path: str, chunk_size: int = 500, chunk_overlap: int = 50) -> dict:
        documents = load_pdf(pdf_path)
        chunks = split_docs(documents, chunk_size, chunk_overlap)
        texts = [c.page_content for c in chunks]
        embeddings = self.embedding_manager.generate_embeddings(texts)
        num_added = self.vector_store.add_documents(chunks, embeddings)
        return {"num_pages": len(documents), "num_chunks": len(chunks), "num_added": num_added}

    def ingest_uploaded_file(self, uploaded_file, chunk_size: int = 500, chunk_overlap: int = 50) -> dict:
        """
        Accepts a Streamlit UploadedFile, writes it to a temp path (PyPDFLoader
        needs a real file path), ingests it, then cleans up.
        """
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name
        try:
            return self.ingest_pdf(tmp_path, chunk_size, chunk_overlap)
        finally:
            os.remove(tmp_path)

    # ---- Query ---------------------------------------------------------
    def query(self, question: str, top_k: int = 5, score_threshold: float = 0.0, chat_history=None) -> dict:
        retrieved_docs = self.retriever.retrieve(question, top_k=top_k, score_threshold=score_threshold)
        if not retrieved_docs:
            return {
                "answer": "I couldn't find relevant content in the uploaded document to answer this question.",
                "sources": [],
            }
        answer = self.generator.generate(question, retrieved_docs, chat_history)
        return {"answer": answer, "sources": retrieved_docs}


# ============================================================
# CLI smoke test: `python app.py path/to/file.pdf "your question"`
# ============================================================
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print('Usage: python app.py <pdf_path> "<question>"')
        raise SystemExit(1)

    pdf_path, question = sys.argv[1], sys.argv[2]
    pipeline = RAGPipeline()
    stats = pipeline.ingest_pdf(pdf_path)
    print(f"Ingested: {stats}")

    result = pipeline.query(question)
    print("\nAnswer:\n", result["answer"])
    print(f"\nSources used: {len(result['sources'])}")
