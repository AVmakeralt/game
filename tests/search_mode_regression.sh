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

CACHE_FILE="${ROOT_DIR}/opening_cache.txt"
CACHE_BACKUP=""
if [[ -f "$CACHE_FILE" ]]; then
  CACHE_BACKUP="$(mktemp)"
  cp "$CACHE_FILE" "$CACHE_BACKUP"
fi
cleanup_cache() {
  rm -f "$CACHE_FILE"
  if [[ -n "$CACHE_BACKUP" && -f "$CACHE_BACKUP" ]]; then
    mv "$CACHE_BACKUP" "$CACHE_FILE"
  fi
}
trap cleanup_cache EXIT
rm -f "$CACHE_FILE"

extract_root_mix() {
  local output="$1"
  local line
  line=$(printf "%s\n" "$output" | awk '/info string eval_breakdown/{print; exit}')
  if [[ ! "$line" =~ root_mix\[ugds=([0-9]+),classic=([0-9]+)\] ]]; then
    echo "missing root_mix search-mode telemetry" >&2
    return 1
  fi
  echo "${BASH_REMATCH[1]} ${BASH_REMATCH[2]}"
}

# Tactical/opening position should use custom UGDS path and never fall back to classic mode.
opening_out=$(cd "$ROOT_DIR" && printf 'position startpos\ngo depth 4\nquit\n' | "$ENGINE")
read -r opening_ugds opening_classic < <(extract_root_mix "$opening_out")
[[ "$opening_ugds" -gt 0 ]]
[[ "$opening_classic" -eq 0 ]]

# Quiet corner-king ending should also stay on the custom UGDS path only.
endgame_out=$(cd "$ROOT_DIR" && printf 'position fen 7k/8/8/8/8/8/8/K7 w - - 0 1\ngo depth 4\nquit\n' | "$ENGINE")
read -r endgame_ugds endgame_classic < <(extract_root_mix "$endgame_out")
[[ "$endgame_ugds" -gt 0 ]]
[[ "$endgame_classic" -eq 0 ]]

# Stress a few representative positions to ensure search does not crash.
for fen in \
  'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1' \
  'r1bq1rk1/ppp2ppp/2np1n2/4p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 8' \
  '8/8/2k5/8/8/5K2/6P1/8 w - - 0 1'
do
  stress_out=$(cd "$ROOT_DIR" && printf 'position fen %s\ngo depth 3\nquit\n' "$fen" | "$ENGINE")
  [[ "$stress_out" == *"bestmove "* ]]
  [[ "$stress_out" != *"Segmentation fault"* ]]
done

echo "search mode regression passed"
