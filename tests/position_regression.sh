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

# Draw-by-fifty-move rule path should trigger immediately in go handling.
out=$(printf 'position fen 7k/8/8/8/8/8/8/K7 w - - 100 1\ngo depth 2\nquit\n' | "$ENGINE")
[[ "$out" == *"info string draw by rule"* ]]
[[ "$out" == *"bestmove 0000"* ]]

echo "position regression passed"
