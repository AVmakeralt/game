#!/usr/bin/env bash
set -euo pipefail

ENGINE="${1:-./chess_engine}"

run_perft() {
  local depth="$1"
  printf "position startpos\nperft ${depth}\nquit\n" | "$ENGINE" | awk '/^nodes/{print $2}'
}

[[ "$(run_perft 1)" == "20" ]]
[[ "$(run_perft 2)" == "400" ]]
[[ "$(run_perft 3)" == "8902" ]]

echo "perft regression passed"
