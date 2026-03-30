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

out=$(cd "$ROOT_DIR" && printf 'weights\nquit\n' | "$ENGINE")
[[ "$out" == *"nnue=eval.nnue"* ]]
[[ "$out" == *"strategy=meganet.lc0"* ]]
[[ "$out" == *"transformer=chess_transformer_25m.pt"* ]]

echo "weights regression passed"
