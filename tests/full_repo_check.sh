#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

pushd "$ROOT_DIR" >/dev/null

cmake -S . -B build
cmake --build build -j"$(nproc)"

for src in board.cpp movegen.cpp eval.cpp tt.cpp main.cpp; do
  c++ -std=c++17 -Wall -Wextra -pedantic -I. -fsyntax-only "$src"
done

for script in tests/perft_regression.sh tests/position_regression.sh tests/search_mode_regression.sh tests/reviewer_regression.sh tests/weights_regression.sh tests/full_repo_check.sh; do
  bash -n "$script"
done

python -m json.tool books/small_book.json >/dev/null
python -m json.tool books/mega_book.json >/dev/null

bash tests/perft_regression.sh
bash tests/position_regression.sh
bash tests/search_mode_regression.sh
bash tests/reviewer_regression.sh
bash tests/weights_regression.sh

popd >/dev/null

echo "full repo check passed"
