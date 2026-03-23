# rag_engine.py
import os
import json
import math
import traceback
from dotenv import load_dotenv
load_dotenv()
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

# =========================================================
# ENV HELPERS
# =========================================================
def env_str(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)).strip())
    except Exception:
        return default


def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)).strip())
    except Exception:
        return default


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name, str(default)).strip().lower()
    return value in {"1", "true", "yes", "y", "on"}


# =========================================================
# CONFIG
# =========================================================
VECTOR_DB_DIR = env_str("VECTOR_DB_DIR", "vectordb")
COLLECTION_NAME = env_str("COLLECTION_NAME", "diabetes_refs")
VERIFIED_FAQ_PATH = env_str("VERIFIED_FAQ_PATH", "data/verified_faq.json")
CHAT_LOG_PATH = env_str("CHAT_LOG_PATH", "logs/chat_history.jsonl")

EMBEDDING_PROVIDER = env_str("EMBEDDING_PROVIDER", "ollama").lower()
LLM_PROVIDER = env_str("LLM_PROVIDER", "ollama").lower()

OLLAMA_BASE_URL = env_str("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_EMBED_MODEL = env_str("OLLAMA_EMBED_MODEL", "nomic-embed-text")
OLLAMA_LLM_MODEL = env_str("OLLAMA_LLM_MODEL", "llama3.1:8b")

OPENAI_API_KEY = env_str("OPENAI_API_KEY", "")
OPENAI_EMBED_MODEL = env_str("OPENAI_EMBED_MODEL", "text-embedding-3-small")
OPENAI_LLM_MODEL = env_str("OPENAI_LLM_MODEL", "gpt-4o-mini")

TOP_K = env_int("TOP_K", 4)
DOC_SIMILARITY_THRESHOLD = env_float("DOC_SIMILARITY_THRESHOLD", 0.35)
FAQ_SIMILARITY_THRESHOLD = env_float("FAQ_SIMILARITY_THRESHOLD", 0.78)
MAX_CONTEXT_CHARS = env_int("MAX_CONTEXT_CHARS", 6000)
MAX_HISTORY_LOG_CHARS = env_int("MAX_HISTORY_LOG_CHARS", 4000)

# =========================================================
# OPTIONAL IMPORTS
# =========================================================
HAS_CHROMA = False
HAS_OLLAMA = False
HAS_OPENAI = False

try:
    from langchain_chroma import Chroma
    HAS_CHROMA = True
except Exception:
    HAS_CHROMA = False

try:
    from langchain_ollama import OllamaEmbeddings, ChatOllama
    HAS_OLLAMA = True
except Exception:
    HAS_OLLAMA = False

try:
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
    HAS_OPENAI = True
except Exception:
    HAS_OPENAI = False


# =========================================================
# GLOBALS
# =========================================================
_embeddings = None
_llm = None
_vectorstore = None

_faq_data: List[Dict[str, Any]] = []
_faq_vectors: List[List[float]] = []


# =========================================================
# BASIC HELPERS
# =========================================================
def _safe_mkdir_for_file(path: str) -> None:
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)


def _truncate(text: str, max_chars: int) -> str:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


def _normalize_space(text: str) -> str:
    return " ".join((text or "").split()).strip()


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0

    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot / (norm_a * norm_b)


# =========================================================
# MODEL INIT
# =========================================================
def get_embeddings():
    global _embeddings

    if _embeddings is not None:
        return _embeddings

    if EMBEDDING_PROVIDER == "ollama":
        if not HAS_OLLAMA:
            raise RuntimeError(
                "EMBEDDING_PROVIDER=ollama tetapi package langchain_ollama belum terpasang. "
                "Install: pip install langchain-ollama"
            )
        _embeddings = OllamaEmbeddings(
            model=OLLAMA_EMBED_MODEL,
            base_url=OLLAMA_BASE_URL,
        )
        return _embeddings

    if EMBEDDING_PROVIDER == "openai":
        if not HAS_OPENAI:
            raise RuntimeError(
                "EMBEDDING_PROVIDER=openai tetapi package langchain_openai belum terpasang. "
                "Install: pip install langchain-openai"
            )
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY belum diisi.")
        _embeddings = OpenAIEmbeddings(
            model=OPENAI_EMBED_MODEL,
            api_key=OPENAI_API_KEY,
        )
        return _embeddings

    raise RuntimeError(f"EMBEDDING_PROVIDER tidak dikenali: {EMBEDDING_PROVIDER}")


def get_llm():
    global _llm

    if _llm is not None:
        return _llm

    if LLM_PROVIDER == "ollama":
        if not HAS_OLLAMA:
            raise RuntimeError(
                "LLM_PROVIDER=ollama tetapi package langchain_ollama belum terpasang. "
                "Install: pip install langchain-ollama"
            )
        _llm = ChatOllama(
            model=OLLAMA_LLM_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0.2,
        )
        return _llm

    if LLM_PROVIDER == "openai":
        if not HAS_OPENAI:
            raise RuntimeError(
                "LLM_PROVIDER=openai tetapi package langchain_openai belum terpasang. "
                "Install: pip install langchain-openai"
            )
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY belum diisi.")
        _llm = ChatOpenAI(
            model=OPENAI_LLM_MODEL,
            api_key=OPENAI_API_KEY,
            temperature=0.2,
        )
        return _llm

    raise RuntimeError(f"LLM_PROVIDER tidak dikenali: {LLM_PROVIDER}")


def get_vectorstore():
    global _vectorstore

    if _vectorstore is not None:
        return _vectorstore

    if not HAS_CHROMA:
        raise RuntimeError(
            "Package langchain_chroma belum terpasang. "
            "Install: pip install langchain-chroma"
        )

    embeddings = get_embeddings()

    _vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=VECTOR_DB_DIR,
    )
    return _vectorstore


# =========================================================
# FAQ LOAD + INDEX
# =========================================================
def load_verified_faq() -> List[Dict[str, Any]]:
    global _faq_data

    if not os.path.exists(VERIFIED_FAQ_PATH):
        print(f"[FAQ] File tidak ditemukan: {VERIFIED_FAQ_PATH}")
        _faq_data = []
        return _faq_data

    try:
        with open(VERIFIED_FAQ_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            print("[FAQ] Format verified_faq.json harus berupa list")
            _faq_data = []
            return _faq_data

        clean_rows = []
        for item in data:
            if not isinstance(item, dict):
                continue

            row = {
                "id": str(item.get("id", "")).strip(),
                "question": str(item.get("question", "")).strip(),
                "answer": str(item.get("answer", "")).strip(),
                "sources": item.get("sources", []) or [],
                "tags": item.get("tags", []) or [],
            }

            if row["question"] and row["answer"]:
                clean_rows.append(row)

        _faq_data = clean_rows
        print(f"[FAQ] Loaded {len(_faq_data)} entries")
        return _faq_data

    except Exception as e:
        print(f"[FAQ] Gagal load FAQ: {e}")
        _faq_data = []
        return _faq_data


def build_verified_faq_index() -> None:
    global _faq_vectors

    embeddings = get_embeddings()

    if not _faq_data:
        load_verified_faq()

    if not _faq_data:
        _faq_vectors = []
        return

    try:
        rows_to_embed = []
        for item in _faq_data:
            tags_text = ", ".join([str(x) for x in item.get("tags", []) if str(x).strip()])
            combined = f"Pertanyaan: {item['question']}\nTag: {tags_text}"
            rows_to_embed.append(combined)

        _faq_vectors = embeddings.embed_documents(rows_to_embed)
        print(f"[FAQ] Built index untuk {len(_faq_vectors)} entries")
    except Exception as e:
        print(f"[FAQ] Gagal build index: {e}")
        _faq_vectors = []


def search_verified_faq(user_question: str) -> Optional[Dict[str, Any]]:
    if not _faq_data:
        load_verified_faq()

    if _faq_data and not _faq_vectors:
        build_verified_faq_index()

    if not _faq_data or not _faq_vectors:
        return None

    try:
        embeddings = get_embeddings()
        query_vec = embeddings.embed_query(user_question)

        best_idx = -1
        best_score = -1.0

        for idx, faq_vec in enumerate(_faq_vectors):
            score = cosine_similarity(query_vec, faq_vec)
            if score > best_score:
                best_idx = idx
                best_score = score

        if best_idx < 0:
            return None

        result = dict(_faq_data[best_idx])
        result["score"] = round(best_score, 4)

        if best_score >= FAQ_SIMILARITY_THRESHOLD:
            return result

        return None

    except Exception as e:
        print(f"[FAQ] Gagal search FAQ: {e}")
        return None


# =========================================================
# SOURCE FORMATTERS
# =========================================================
def _source_label_from_metadata(meta: Dict[str, Any]) -> str:
    if not meta:
        return "Referensi"

    date = str(meta.get("date", "")).strip()
    org = str(meta.get("organization", "")).strip()
    title = str(meta.get("title", "")).strip()
    year = str(meta.get("year", "")).strip()
    source = str(meta.get("source", "")).strip()
    filename = str(meta.get("filename", "")).strip()

    parts = []

    if date:
        parts.append(date)
    elif year:
        parts.append(year)

    if org:
        parts.append(org)

    if title:
        parts.append(title)
    elif source:
        parts.append(source)
    elif filename:
        parts.append(filename)

    label = " - ".join([p for p in parts if p])
    return label if label else "Referensi"


def format_sources_from_docs(docs: List[Any]) -> List[str]:
    seen = set()
    results = []

    for doc in docs or []:
        meta = getattr(doc, "metadata", {}) or {}
        label = _source_label_from_metadata(meta)
        if label not in seen:
            seen.add(label)
            results.append(label)

    return results


def format_sources_block(sources: List[str]) -> str:
    clean = []
    seen = set()

    for s in sources or []:
        s = str(s).strip()
        if s and s not in seen:
            seen.add(s)
            clean.append(s)

    if not clean:
        return ""

    lines = "\n".join([f"• {x}" for x in clean])
    return f"\n\nSumber Referensi:\n{lines}"


# =========================================================
# ANSWER RENDERERS
# =========================================================
def render_final_answer(answer: str, sources: List[str]) -> str:
    answer = (answer or "").strip()

    if not answer:
        answer = "Maaf, saya belum bisa menyusun jawaban yang memadai."

    disclaimer = (
        "\n\nCatatan:\n"
        "Informasi ini bersifat edukasi dan tidak menggantikan konsultasi medis langsung. "
        "Untuk keputusan medis, perubahan obat, perubahan dosis insulin, atau kondisi darurat, tetap konsultasikan ke dokter."
    )

    return f"{answer}{format_sources_block(sources)}{disclaimer}"


def render_safe_answer() -> str:
    return (
        "Maaf, saya belum menemukan jawaban yang cukup kuat dari referensi yang tersedia untuk pertanyaan ini.\n\n"
        "Untuk pertanyaan tentang dosis obat, insulin, interpretasi hasil lab, atau kondisi yang terasa darurat, sebaiknya langsung konsultasikan ke dokter.\n\n"
        "Catatan:\n"
        "Informasi ini bersifat edukasi dan tidak menggantikan konsultasi medis langsung."
    )


# =========================================================
# PROMPT
# =========================================================
def build_prompt(context: str, question: str) -> str:
    return f"""
Anda adalah asisten edukasi diabetes berbahasa Indonesia.

Tugas Anda:
1. Jawab pertanyaan pengguna HANYA berdasarkan konteks referensi yang diberikan.
2. Gunakan bahasa yang sederhana, singkat, dan mudah dipahami orang awam.
3. Jangan menyebut istilah seperti "dokumen internal", "fallback", "retrieval", "mode", "sumber internal", atau "sumber eksternal".
4. Jangan mengarang diagnosis, jangan mengarang dosis, dan jangan membuat keputusan medis final.
5. Jika konteks tidak cukup, katakan dengan jujur bahwa referensi yang tersedia belum cukup kuat.
6. Fokus pada jawaban yang praktis dan aman.

Konteks referensi:
{context}

Pertanyaan pengguna:
{question}

Jawaban:
""".strip()


# =========================================================
# LOGGING
# =========================================================
def log_chat_interaction(
    question: str,
    answer: str,
    mode: str,
    sources: Optional[List[str]] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    try:
        _safe_mkdir_for_file(CHAT_LOG_PATH)

        row = {
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "answer": _truncate(answer, MAX_HISTORY_LOG_CHARS),
            "mode": mode,
            "sources": sources or [],
            "meta": meta or {},
        }

        with open(CHAT_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    except Exception as e:
        print(f"[LOG] Gagal simpan log: {e}")


# =========================================================
# INTERNAL SEARCH
# =========================================================
def search_internal_docs(question: str, top_k: int = TOP_K) -> Dict[str, Any]:
    try:
        vectorstore = get_vectorstore()

        results = vectorstore.similarity_search_with_relevance_scores(question, k=top_k)

        docs = []
        scores = []

        for item in results:
            if isinstance(item, tuple) and len(item) >= 2:
                doc, score = item[0], item[1]
                docs.append(doc)
                scores.append(float(score))

        if not docs:
            return {
                "confident": False,
                "docs": [],
                "scores": [],
                "avg_score": 0.0,
                "sources": [],
                "context": "",
            }

        avg_score = sum(scores) / len(scores) if scores else 0.0
        confident = avg_score >= DOC_SIMILARITY_THRESHOLD

        context_chunks = []
        for i, doc in enumerate(docs, start=1):
            text = getattr(doc, "page_content", "") or ""
            text = _normalize_space(text)
            if not text:
                continue

            src = _source_label_from_metadata(getattr(doc, "metadata", {}) or {})
            chunk = f"[Referensi {i}] {src}\n{text}"
            context_chunks.append(chunk)

        context = "\n\n".join(context_chunks)
        context = _truncate(context, MAX_CONTEXT_CHARS)

        return {
            "confident": confident,
            "docs": docs,
            "scores": scores,
            "avg_score": round(avg_score, 4),
            "sources": format_sources_from_docs(docs),
            "context": context,
        }

    except Exception as e:
        print(f"[RAG] Gagal search internal docs: {e}")
        traceback.print_exc()
        return {
            "confident": False,
            "docs": [],
            "scores": [],
            "avg_score": 0.0,
            "sources": [],
            "context": "",
        }


# =========================================================
# LLM ANSWER
# =========================================================
def generate_answer_from_context(context: str, question: str) -> str:
    try:
        if not context.strip():
            return ""

        llm = get_llm()
        prompt = build_prompt(context=context, question=question)
        response = llm.invoke(prompt)

        content = getattr(response, "content", None)
        if isinstance(content, str) and content.strip():
            return content.strip()

        return str(response).strip()

    except Exception as e:
        print(f"[LLM] Gagal generate answer: {e}")
        traceback.print_exc()
        return ""


# =========================================================
# MAIN PIPELINE
# =========================================================
def answer_question(question: str) -> Tuple[str, str, List[str]]:
    question = (question or "").strip()

    if not question:
        answer = (
            "Silakan kirim pertanyaan yang ingin Anda tanyakan tentang diabetes.\n\n"
            "Contoh:\n"
            "• Berapa target HbA1c pada diabetes tipe 2?\n"
            "• Apa tanda hipoglikemia?\n"
            "• Makanan apa yang baik untuk diabetes tipe 2?"
        )
        return answer, "empty", []

    internal_result = search_internal_docs(question)

    if internal_result.get("confident"):
        raw_answer = generate_answer_from_context(
            context=internal_result.get("context", ""),
            question=question,
        )

        if raw_answer.strip():
            final_answer = render_final_answer(
                answer=raw_answer,
                sources=internal_result.get("sources", []),
            )
            return final_answer, "docs", internal_result.get("sources", [])

    faq_result = search_verified_faq(question)
    if faq_result:
        final_answer = render_final_answer(
            answer=faq_result.get("answer", ""),
            sources=faq_result.get("sources", []),
        )
        return final_answer, "faq", faq_result.get("sources", [])

    final_answer = render_safe_answer()
    return final_answer, "safe", []


def ask_and_render(question: str) -> str:
    try:
        final_answer, mode, sources = answer_question(question)

        log_chat_interaction(
            question=question,
            answer=final_answer,
            mode=mode,
            sources=sources,
        )

        return final_answer

    except Exception as e:
        traceback.print_exc()

        fallback = (
            "Maaf, terjadi kendala saat memproses pertanyaan Anda.\n\n"
            "Silakan coba lagi. Jika pertanyaannya berkaitan dengan obat, insulin, atau kondisi darurat, sebaiknya langsung hubungi dokter."
        )

        log_chat_interaction(
            question=question,
            answer=fallback,
            mode="error",
            sources=[],
            meta={"error": str(e)},
        )

        return fallback


# =========================================================
# STARTUP
# =========================================================
def initialize_knowledge() -> None:
    try:
        print("[INIT] Memulai inisialisasi knowledge...")

        get_embeddings()

        try:
            get_vectorstore()
            print("[INIT] Vector DB siap")
        except Exception as e:
            print(f"[INIT] Vector DB belum siap: {e}")

        load_verified_faq()
        if _faq_data:
            build_verified_faq_index()

        print("[INIT] Selesai")
    except Exception as e:
        print(f"[INIT] Gagal inisialisasi: {e}")


# =========================================================
# TEST CLI
# =========================================================
if __name__ == "__main__":
    initialize_knowledge()

    print("Diabetes RAG Engine siap. Ketik 'exit' untuk keluar.\n")

    while True:
        q = input("Anda: ").strip()
        if q.lower() in {"exit", "quit"}:
            break

        result = ask_and_render(q)
        print("\nJawaban:\n")
        print(result)
        print("\n" + "=" * 80 + "\n")