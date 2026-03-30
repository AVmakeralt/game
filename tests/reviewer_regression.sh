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

out=$(cd "$ROOT_DIR" && printf 'uci\nsetoption name ReviewerEnabled value true\nsetoption name ReviewerWeights value reviewer_topics.nn\nweights\nreviewgame 1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Bxc6 dxc6 5. O-O O-O\npracticetopics\nquit\n' | "$ENGINE")

[[ "$out" == *"reviewer=reviewer_topics.nn reviewer_enabled=true"* ]]
[[ "$out" == *"info string practice_topics "* ]]
[[ "$out" == *"opening-preparation"* ]]
[[ "$out" == *"tactics-and-calculation"* ]]

echo "reviewer regression passed"
