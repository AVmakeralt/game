"""Opening book builder for SmallNet and MegaNet.

Generates deterministic JSON books keyed by move-history strings using the
engine's `openingKey` convention (`startpos` or `move_move_...`).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


Book = Dict[str, str]


def _add_line(book: Book, line: List[str]) -> None:
    key = "startpos"
    for move in line:
        if key not in book:
            book[key] = move
        key = move if key == "startpos" else f"{key}_{move}"


def build_small_book() -> Book:
    # Fast, solid repertoire for lightweight model.
    lines = [
        ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5"],  # Ruy Lopez
        ["e2e4", "c7c5", "g1f3", "d7d6", "d2d4"],  # Open Sicilian
        ["d2d4", "d7d5", "c2c4", "e7e6", "b1c3"],  # QGD setup
        ["c2c4", "e7e5", "b1c3", "g8f6", "g2g3"],  # English
        ["g1f3", "d7d5", "g2g3", "g8f6", "f1g2"],  # Réti/King's Fianchetto
    ]
    book: Book = {}
    for line in lines:
        _add_line(book, line)
    return book


def build_mega_book() -> Book:
    # Larger repertoire with sharper branches for stronger net.
    lines = [
        ["e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4", "g8f6", "b1c3"],
        ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4", "g8f6", "e1g1"],
        ["d2d4", "g8f6", "c2c4", "g7g6", "b1c3", "f8g7", "e2e4", "d7d6", "g1f3"],
        ["d2d4", "d7d5", "c2c4", "e7e6", "b1c3", "g8f6", "c1g5", "f8e7", "e2e3"],
        ["c2c4", "e7e5", "b1c3", "g8f6", "g2g3", "d7d5", "c4d5", "f6d5", "f1g2"],
        ["e2e4", "c7c6", "d2d4", "d7d5", "b1c3", "d5e4", "c3e4", "c8f5", "e4g3"],
    ]
    book: Book = {}
    for line in lines:
        _add_line(book, line)
    return book


def write_book(path: Path, book: Book) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(book, f, indent=2, sort_keys=True)
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build opening books for SmallNet and MegaNet.")
    parser.add_argument("--out-dir", type=Path, default=Path("books"))
    args = parser.parse_args()

    small = build_small_book()
    mega = build_mega_book()
    write_book(args.out_dir / "small_book.json", small)
    write_book(args.out_dir / "mega_book.json", mega)
    print(f"wrote small_book.json entries={len(small)}")
    print(f"wrote mega_book.json entries={len(mega)}")


if __name__ == "__main__":
    main()
