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

check_bestmove_is_legal() {
  local fen="$1"
  local best
  best=$(cd "$ROOT_DIR" && printf 'position fen %s\ngo depth 3\nquit\n' "$fen" | "$ENGINE" | awk '/^bestmove /{print $2; exit}')
  [[ -n "$best" ]]
  [[ "$best" != "0000" ]]

  local legal_out
  legal_out=$(cd "$ROOT_DIR" && printf 'position fen %s\nislegal %s\nquit\n' "$fen" "$best" | "$ENGINE")
  [[ "$legal_out" == *"info string legal true"* ]]
}

check_bestmove_is_legal '4k3/8/8/8/8/8/4r3/4K3 w - - 0 1'
check_bestmove_is_legal 'r3k2r/pppq1ppp/2n1bn2/3pp3/3PP3/2N1BN2/PPPQ1PPP/R3K2R w KQkq - 0 1'

echo "legal move regression passed"
