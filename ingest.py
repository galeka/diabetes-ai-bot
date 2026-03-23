from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import fitz  # pymupdf
import pytesseract
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from PIL import Image

try:
    from langchain_chroma import Chroma
except Exception:
    from langchain_community.vectorstores import Chroma


load_dotenv()

# =========================
# Config
# =========================
REFERENCES_DIR = Path(os.getenv("REFERENCES_DIR", "references"))
VECTOR_DB_DIR = Path(os.getenv("VECTOR_DB_DIR", "vectordb"))
COLLECTION_NAME = os.getenv("VECTOR_COLLECTION_NAME", "diabetes_refs")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "900"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "180"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "500"))
RETRY_COUNT = int(os.getenv("INGEST_RETRY_COUNT", "3"))
RETRY_DELAY_SEC = float(os.getenv("INGEST_RETRY_DELAY_SEC", "2"))
MIN_PAGE_TEXT_LEN = int(os.getenv("MIN_PAGE_TEXT_LEN", "40"))

ENABLE_OCR_FALLBACK = os.getenv("ENABLE_OCR_FALLBACK", "true").lower() == "true"
OCR_LANG = os.getenv("OCR_LANG", "eng")
OCR_DPI = int(os.getenv("OCR_DPI", "200"))
OCR_TEXT_MIN_LEN = int(os.getenv("OCR_TEXT_MIN_LEN", "20"))
TESSERACT_CMD = os.getenv("TESSERACT_CMD", "").strip()

STRICT_DIABETES_ONLY = os.getenv("STRICT_DIABETES_ONLY", "false").lower() == "true"
SKIP_EMPTY_PDF_AFTER_OCR = os.getenv("SKIP_EMPTY_PDF_AFTER_OCR", "true").lower() == "true"

MANIFEST_PATH = VECTOR_DB_DIR / "index_manifest.json"
LOCK_PATH = VECTOR_DB_DIR / ".ingest.lock"


# =========================
# Utility
# =========================
def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")



def elapsed_str(seconds: float) -> str:
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h}j {m}m {s}d"
    if m > 0:
        return f"{m}m {s}d"
    return f"{s}d"



def normalize_text(text: str) -> str:
    text = (text or "").replace("\x00", " ")
    text = text.replace("\ufeff", " ")
    text = text.replace("•", " ")
    text = text.replace("\t", " ")
    text = re.sub(r"[ ]{2,}", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text



def looks_like_noise(text: str) -> bool:
    stripped = (text or "").strip()
    if len(stripped) < MIN_PAGE_TEXT_LEN:
        return True

    alnum_count = sum(c.isalnum() for c in stripped)
    if alnum_count == 0:
        return True

    alpha_ratio = sum(c.isalpha() for c in stripped) / max(len(stripped), 1)
    return alpha_ratio < 0.2



def file_hash(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()



def infer_topic(filename: str) -> str:
    name = filename.lower()
    rules = {
        "insulin": ["insulin"],
        "hipoglikemia": ["hipoglikemia", "hypoglycemia", "gula rendah", "drop"],
        "hiperglikemia": ["hiperglikemia", "hyperglycemia", "gula tinggi"],
        "ramadan": ["ramadan", "puasa"],
        "kehamilan": ["kehamilan", "pregnancy", "hamil"],
        "dislipidemia": ["dislipidemia", "lipid", "kolesterol"],
        "tb_dm": ["tb-dm", "tb dm", "tuberkulosis"],
        "monitoring": ["pemantauan", "monitor", "glukosa darah mandiri"],
        "type_1": ["type 1", "tipe 1"],
        "type_2": ["type 2", "tipe 2", "dmt2", "dm tipe 2"],
        "hospital": ["rumah sakit", "hospital", "rawat inap"],
        "general_diabetes": ["diabetes", "dm", "hba1c", "glucose", "glycemic", "glycaemic"],
    }
    for topic, keywords in rules.items():
        if any(keyword in name for keyword in keywords):
            return topic
    return "general"



def is_relevant_diabetes_file(filename: str) -> bool:
    name = filename.lower()
    keywords = [
        "diabetes", "dm", "hba1c", "glucose", "glycemic", "glycaemic",
        "insulin", "metformin", "hypoglycemia", "hyperglycemia",
        "hipoglikemia", "hiperglikemia", "endocrine", "endokrin",
    ]
    return any(keyword in name for keyword in keywords)



def ensure_dirs() -> None:
    REFERENCES_DIR.mkdir(parents=True, exist_ok=True)
    VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)



def acquire_lock() -> None:
    if LOCK_PATH.exists():
        raise RuntimeError(
            f"Lock file ditemukan: {LOCK_PATH}. "
            "Kemungkinan ingest lain masih berjalan atau proses sebelumnya crash."
        )
    LOCK_PATH.write_text(f"locked_at={now_str()}\n", encoding="utf-8")



def release_lock() -> None:
    if LOCK_PATH.exists():
        LOCK_PATH.unlink(missing_ok=True)



def load_manifest() -> dict[str, Any]:
    if not MANIFEST_PATH.exists():
        return {"version": 4, "updated_at": None, "files": {}}
    try:
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except Exception:
        print("⚠ Manifest rusak. Membuat manifest baru.")
        return {"version": 4, "updated_at": None, "files": {}}



def save_manifest(manifest: dict[str, Any]) -> None:
    manifest["updated_at"] = now_str()
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")



def recreate_vector_store_dir() -> None:
    if VECTOR_DB_DIR.exists():
        shutil.rmtree(VECTOR_DB_DIR)
    VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)



def get_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_BASE_URL)



def get_db() -> Chroma:
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(VECTOR_DB_DIR),
        embedding_function=get_embeddings(),
    )



def configure_tesseract() -> None:
    if TESSERACT_CMD:
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD



def render_pdf_page_for_ocr(pdf_path: Path, page_index: int, dpi: int = OCR_DPI) -> Image.Image:
    doc = fitz.open(pdf_path)
    try:
        page = doc.load_page(page_index)
        zoom = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        mode = "RGB" if pix.n < 4 else "RGBA"
        image = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
        return image.convert("RGB")
    finally:
        doc.close()



def extract_page_text_with_ocr(pdf_path: Path, page_index: int) -> str:
    image = render_pdf_page_for_ocr(pdf_path, page_index, dpi=OCR_DPI)
    text = pytesseract.image_to_string(image, lang=OCR_LANG)
    return normalize_text(text)



def load_clean_pages(pdf_file: Path) -> tuple[list[Document], dict[str, Any]]:
    print(f"\nLoading: {pdf_file.name}")

    if STRICT_DIABETES_ONLY and not is_relevant_diabetes_file(pdf_file.name):
        print("   ⚠ Skip file non-diabetes")
        return [], {
            "file_name": pdf_file.name,
            "file_hash": file_hash(pdf_file),
            "topic": infer_topic(pdf_file.name),
            "loaded_pages_raw": 0,
            "loaded_pages_clean": 0,
            "skipped_short_pages": 0,
            "ocr_used_pages": 0,
            "raw_empty_pages": 0,
            "skipped_non_diabetes": True,
        }

    loader = PyPDFLoader(str(pdf_file))
    pages = loader.load()

    topic = infer_topic(pdf_file.name)
    current_file_hash = file_hash(pdf_file)

    clean_pages: list[Document] = []
    skipped_short = 0
    ocr_used_pages = 0
    raw_empty_pages = 0

    for idx, page in enumerate(pages):
        raw_text = normalize_text(page.page_content)
        if not raw_text:
            raw_empty_pages += 1

        final_text = raw_text
        extraction_method = "text"

        if looks_like_noise(raw_text) and ENABLE_OCR_FALLBACK:
            try:
                ocr_text = extract_page_text_with_ocr(pdf_file, idx)
                if len(ocr_text) >= OCR_TEXT_MIN_LEN and len(ocr_text) > len(raw_text):
                    final_text = ocr_text
                    extraction_method = "ocr"
                    ocr_used_pages += 1
            except Exception as exc:
                print(f"   ⚠ OCR gagal di page {idx + 1}: {exc}")

        final_text = normalize_text(final_text)
        if looks_like_noise(final_text):
            skipped_short += 1
            continue

        metadata = dict(page.metadata or {})
        metadata["source_file"] = pdf_file.name
        metadata["source_path"] = str(pdf_file.resolve())
        metadata["topic"] = topic
        metadata["file_hash"] = current_file_hash
        metadata["page"] = metadata.get("page", idx)
        metadata["extraction_method"] = extraction_method

        clean_pages.append(Document(page_content=final_text, metadata=metadata))

    stats = {
        "file_name": pdf_file.name,
        "file_hash": current_file_hash,
        "topic": topic,
        "loaded_pages_raw": len(pages),
        "loaded_pages_clean": len(clean_pages),
        "skipped_short_pages": skipped_short,
        "ocr_used_pages": ocr_used_pages,
        "raw_empty_pages": raw_empty_pages,
        "skipped_non_diabetes": False,
    }

    print(f"   Loaded pages raw    : {len(pages)}")
    print(f"   Loaded pages bersih : {len(clean_pages)}")
    print(f"   OCR pages used      : {ocr_used_pages}")
    print(f"   Raw empty pages     : {raw_empty_pages}")
    print(f"   Skipped noisy pages : {skipped_short}")

    return clean_pages, stats



def build_chunks(pages: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],
        add_start_index=True,
    )
    return splitter.split_documents(pages)



def build_chunk_ids(chunks: list[Document], file_hash_value: str) -> list[str]:
    ids: list[str] = []
    for idx, chunk in enumerate(chunks):
        page = chunk.metadata.get("page", 0)
        start_index = chunk.metadata.get("start_index", idx)
        raw = f"{file_hash_value}:{page}:{start_index}:{idx}"
        stable_id = hashlib.md5(raw.encode("utf-8")).hexdigest()
        ids.append(stable_id)
    return ids



def delete_ids_in_batches(db: Chroma, ids: list[str], batch_size: int = BATCH_SIZE) -> int:
    if not ids:
        return 0

    deleted = 0
    total_batches = (len(ids) + batch_size - 1) // batch_size
    for i in range(0, len(ids), batch_size):
        batch = ids[i : i + batch_size]
        batch_num = (i // batch_size) + 1
        print(f"   Hapus batch lama {batch_num}/{total_batches} ({len(batch)} IDs)...")
        db.delete(ids=batch)
        deleted += len(batch)
    return deleted



def add_documents_with_retry(
    db: Chroma,
    docs: list[Document],
    ids: list[str],
    batch_size: int = BATCH_SIZE,
    retry_count: int = RETRY_COUNT,
    retry_delay_sec: float = RETRY_DELAY_SEC,
    started_at: float | None = None,
) -> int:
    total = len(docs)
    if total == 0:
        return 0

    total_batches = (total + batch_size - 1) // batch_size
    inserted = 0

    for i in range(0, total, batch_size):
        batch_docs = docs[i : i + batch_size]
        batch_ids = ids[i : i + batch_size]
        batch_num = (i // batch_size) + 1

        attempt = 0
        while True:
            attempt += 1
            try:
                db.add_documents(batch_docs, ids=batch_ids)
                inserted += len(batch_docs)
                elapsed = elapsed_str(time.time() - started_at) if started_at else "-"
                print(
                    f"   Batch {batch_num}/{total_batches} sukses "
                    f"({len(batch_docs)} chunks). Total inserted: {inserted}/{total}. "
                    f"Elapsed: {elapsed}"
                )
                break
            except Exception as exc:
                if attempt >= retry_count:
                    print(f"   ❌ Batch {batch_num}/{total_batches} gagal total.")
                    raise RuntimeError(
                        f"Gagal insert batch {batch_num}/{total_batches} setelah {retry_count} percobaan. Error: {exc}"
                    ) from exc

                print(
                    f"   ⚠ Batch {batch_num}/{total_batches} gagal "
                    f"(attempt {attempt}/{retry_count}). Retry {retry_delay_sec} detik..."
                )
                time.sleep(retry_delay_sec)

    return inserted



def plan_files(pdf_files: list[Path], manifest: dict[str, Any]) -> dict[str, list[Path]]:
    files_state = manifest.get("files", {})
    new_files: list[Path] = []
    changed_files: list[Path] = []
    unchanged_files: list[Path] = []

    for pdf_file in pdf_files:
        current_hash = file_hash(pdf_file)
        existing = files_state.get(pdf_file.name)

        if not existing:
            new_files.append(pdf_file)
        elif existing.get("file_hash") != current_hash:
            changed_files.append(pdf_file)
        else:
            unchanged_files.append(pdf_file)

    return {"new": new_files, "changed": changed_files, "unchanged": unchanged_files}



def cleanup_deleted_files(db: Chroma, manifest: dict[str, Any], existing_pdf_names: set[str]) -> int:
    files_state = manifest.get("files", {})
    stale_names = [name for name in files_state.keys() if name not in existing_pdf_names]
    removed_count = 0

    if not stale_names:
        return 0

    print("\nMembersihkan file yang sudah tidak ada di folder references...")
    for name in stale_names:
        entry = files_state.get(name, {})
        old_ids = entry.get("chunk_ids", [])
        if old_ids:
            print(f" - Menghapus chunk lama untuk file hilang: {name}")
            delete_ids_in_batches(db, old_ids, batch_size=BATCH_SIZE)
        files_state.pop(name, None)
        removed_count += 1

    return removed_count



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Incremental PDF ingest to Chroma with OCR fallback")
    parser.add_argument("--full-rebuild", action="store_true", help="Hapus vector DB dan manifest, lalu index ulang semua file.")
    parser.add_argument("--dry-run", action="store_true", help="Tampilkan rencana ingest tanpa menulis ke DB.")
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    started_at = time.time()
    ensure_dirs()
    configure_tesseract()

    try:
        acquire_lock()
    except Exception as exc:
        print(f"❌ {exc}")
        sys.exit(1)

    summary = {
        "pdf_found": 0,
        "new_files": 0,
        "changed_files": 0,
        "unchanged_files": 0,
        "deleted_missing_files": 0,
        "processed_files": 0,
        "skipped_files": 0,
        "empty_clean_files": 0,
        "total_pages_clean": 0,
        "total_chunks": 0,
        "total_deleted_chunks": 0,
        "total_inserted_chunks": 0,
        "total_ocr_pages": 0,
    }

    try:
        print("Mulai indexing...")
        print(f"Waktu mulai: {now_str()}")
        print(f"References dir: {REFERENCES_DIR.resolve()}")
        print(f"Vector DB dir: {VECTOR_DB_DIR.resolve()}")
        print(f"Collection name: {COLLECTION_NAME}")
        print(f"Model embedding: {EMBED_MODEL}")
        print(f"Mode: {'FULL REBUILD' if args.full_rebuild else 'INCREMENTAL'}")
        print(f"Dry run: {'YA' if args.dry_run else 'TIDAK'}")
        print(f"OCR fallback: {'AKTIF' if ENABLE_OCR_FALLBACK else 'NONAKTIF'}")
        print(f"Strict diabetes only: {'YA' if STRICT_DIABETES_ONLY else 'TIDAK'}")

        pdf_files = sorted(REFERENCES_DIR.glob("*.pdf"))
        summary["pdf_found"] = len(pdf_files)

        if not pdf_files:
            print("Tidak ada file PDF di folder references.")
            return

        if args.full_rebuild:
            print("\n⚠ Full rebuild aktif. Menghapus vector DB lama...")
            if not args.dry_run:
                recreate_vector_store_dir()
            manifest = {"version": 4, "updated_at": None, "files": {}}
        else:
            manifest = load_manifest()

        db = get_db() if not args.dry_run else None

        existing_pdf_names = {path.name for path in pdf_files}
        if db is not None:
            summary["deleted_missing_files"] = cleanup_deleted_files(db, manifest, existing_pdf_names)

        plan = plan_files(pdf_files, manifest)
        summary["new_files"] = len(plan["new"])
        summary["changed_files"] = len(plan["changed"])
        summary["unchanged_files"] = len(plan["unchanged"])
        summary["skipped_files"] = len(plan["unchanged"])

        print("\nRencana ingest:")
        print(f" - File baru     : {summary['new_files']}")
        print(f" - File berubah  : {summary['changed_files']}")
        print(f" - File skip     : {summary['unchanged_files']}")

        if args.dry_run:
            print("\nDry run selesai. Tidak ada perubahan ke DB.")
            return

        files_to_process = plan["new"] + plan["changed"]
        files_state = manifest.setdefault("files", {})

        if not files_to_process:
            print("\nTidak ada file baru atau berubah. DB sudah up to date.")
            save_manifest(manifest)
            return

        for file_index, pdf_file in enumerate(files_to_process, start=1):
            print("\n" + "=" * 60)
            print(f"Memproses file {file_index}/{len(files_to_process)}: {pdf_file.name}")

            existing_entry = files_state.get(pdf_file.name)
            if existing_entry:
                old_ids = existing_entry.get("chunk_ids", [])
                if old_ids:
                    print(f"   File berubah. Menghapus {len(old_ids)} chunk lama...")
                    deleted = delete_ids_in_batches(db, old_ids, batch_size=BATCH_SIZE)
                    summary["total_deleted_chunks"] += deleted

            pages, page_stats = load_clean_pages(pdf_file)

            if page_stats.get("skipped_non_diabetes"):
                summary["skipped_files"] += 1
                continue

            summary["total_pages_clean"] += page_stats["loaded_pages_clean"]
            summary["total_ocr_pages"] += page_stats["ocr_used_pages"]

            if not pages:
                print(f"   ⚠ Tidak ada halaman bersih untuk {pdf_file.name}. Skip.")
                summary["empty_clean_files"] += 1
                if SKIP_EMPTY_PDF_AFTER_OCR:
                    files_state[pdf_file.name] = {
                        "file_name": pdf_file.name,
                        "file_hash": page_stats["file_hash"],
                        "topic": page_stats["topic"],
                        "indexed_at": now_str(),
                        "loaded_pages_raw": page_stats["loaded_pages_raw"],
                        "loaded_pages_clean": 0,
                        "ocr_used_pages": page_stats["ocr_used_pages"],
                        "chunk_count": 0,
                        "chunk_ids": [],
                        "status": "empty_clean_text",
                    }
                    save_manifest(manifest)
                continue

            chunks = build_chunks(pages)
            chunk_ids = build_chunk_ids(chunks, page_stats["file_hash"])
            summary["total_chunks"] += len(chunks)

            print(f"   Total chunks file ini: {len(chunks)}")
            inserted = add_documents_with_retry(
                db=db,
                docs=chunks,
                ids=chunk_ids,
                batch_size=BATCH_SIZE,
                retry_count=RETRY_COUNT,
                retry_delay_sec=RETRY_DELAY_SEC,
                started_at=started_at,
            )

            summary["total_inserted_chunks"] += inserted
            summary["processed_files"] += 1

            files_state[pdf_file.name] = {
                "file_name": pdf_file.name,
                "file_hash": page_stats["file_hash"],
                "topic": page_stats["topic"],
                "indexed_at": now_str(),
                "loaded_pages_raw": page_stats["loaded_pages_raw"],
                "loaded_pages_clean": page_stats["loaded_pages_clean"],
                "ocr_used_pages": page_stats["ocr_used_pages"],
                "chunk_count": len(chunks),
                "chunk_ids": chunk_ids,
                "status": "indexed",
            }
            save_manifest(manifest)

        print("\n✅ Selesai indexing")
        print(f"Vector DB: {VECTOR_DB_DIR.resolve()}")

    except Exception as exc:
        print(f"\n❌ Gagal saat ingest: {exc}")
        raise
    finally:
        release_lock()
        total_elapsed = time.time() - started_at
        print("\n" + "=" * 60)
        print("RINGKASAN")
        print("=" * 60)
        print(f"PDF ditemukan           : {summary['pdf_found']}")
        print(f"File baru               : {summary['new_files']}")
        print(f"File berubah            : {summary['changed_files']}")
        print(f"File skip               : {summary['skipped_files']}")
        print(f"File kosong bersih      : {summary['empty_clean_files']}")
        print(f"File hilang dibersihkan : {summary['deleted_missing_files']}")
        print(f"File diproses           : {summary['processed_files']}")
        print(f"Total halaman bersih    : {summary['total_pages_clean']}")
        print(f"Total OCR pages         : {summary['total_ocr_pages']}")
        print(f"Total chunks dibuat     : {summary['total_chunks']}")
        print(f"Total chunks dihapus    : {summary['total_deleted_chunks']}")
        print(f"Total chunks diinsert   : {summary['total_inserted_chunks']}")
        print(f"Durasi total            : {elapsed_str(total_elapsed)}")
        print("=" * 60)


if __name__ == "__main__":
    main()
