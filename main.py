from __future__ import annotations

from src.search import FileSearchResult, search_similar_files


DEFAULT_K = 8
DEFAULT_SCORE_THRESHOLD = 0.2


def print_results(matches: list[FileSearchResult]) -> None:
    if not matches:
        print("\nNo matching files found.")
        return

    print("\nSimilar files:")
    for idx, match in enumerate(matches, start=1):
        print(f"{idx}. {match.file_name}")
        print(f"   location: {match.source_path}")
        print(f"   similarity: {match.score:.4f}")


def main() -> None:
    print("Semantic File Search")
    print("Enter a prompt to find similar files. Type 'q' to quit.\n")

    while True:
        prompt = input("search> ").strip()
        if prompt.lower() in {"q", "quit", "exit"}:
            print("Bye.")
            return
        if not prompt:
            continue

        try:
            matches = search_similar_files(
                prompt,
                k=DEFAULT_K,
                score_threshold=DEFAULT_SCORE_THRESHOLD,
            )
        except Exception as exc:
            print(f"\nSearch failed: {exc}\n")
            continue

        print_results(matches)
        print()


if __name__ == "__main__":
    main()
