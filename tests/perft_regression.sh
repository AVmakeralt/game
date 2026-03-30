#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENGINE="${1:-${ROOT_DIR}/chess_engine}"
if [[ ! -x "$ENGINE" && -x "${ROOT_DIR}/build/chess_engine" ]]; then
  ENGINE="${ROOT_DIR}/build/chess_engine"
fi

if [[ ! -x "$ENGINE" ]]; then
  echo "engine executable not found: $ENGINE" >&2
  exit 1
fi

run_perft() {
  local depth="$1"
  printf "position startpos\nperft ${depth}\nquit\n" | "$ENGINE" | awk '/^nodes/{print $2}'
}

[[ "$(run_perft 1)" == "20" ]]
[[ "$(run_perft 2)" == "400" ]]
[[ "$(run_perft 3)" == "8902" ]]

echo "perft regression passed"
