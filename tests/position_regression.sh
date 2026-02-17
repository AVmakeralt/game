#!/usr/bin/env bash
set -euo pipefail

ENGINE="${1:-./chess_engine}"

# Draw-by-fifty-move rule path should trigger immediately in go handling.
out=$(printf 'position fen 7k/8/8/8/8/8/8/K7 w - - 100 1\ngo depth 2\nquit\n' | "$ENGINE")
[[ "$out" == *"info string draw by rule"* ]]
[[ "$out" == *"bestmove 0000"* ]]

echo "position regression passed"
