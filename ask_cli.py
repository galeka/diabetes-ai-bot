from __future__ import annotations

from rag_engine import answer_question


def print_sources(result: dict) -> None:
    sources = result.get("sources") or []
    if not sources:
        return

    print("\nSumber:")
    seen = set()
    for source in sources:
        label = source.get("organization") or source.get("source_file") or "Sumber ilmiah"
        year = source.get("year")
        key = (label, year)
        if key in seen:
            continue
        seen.add(key)
        if year:
            print(f"• {label}, {year}")
        else:
            print(f"• {label}")


def main() -> None:
    print("=== Diabetes Safe CLI ===")
    print("Ketik 'exit' untuk keluar.\n")

    while True:
        question = input("Pertanyaan: ").strip()
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break

        result = answer_question(question)

        print("\n" + "=" * 100)
        print("JAWABAN")
        print("=" * 100)
        print(result["answer"])
        print_sources(result)
        print("\n")


if __name__ == "__main__":
    main()
