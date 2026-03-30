# Game Chess Engine (C++)

A chess engine with legal move generation, alpha-beta/PVS style search, quiescence, transposition table, and perft verification commands.

## Build

### g++
```bash
g++ -std=c++17 -O2 -Wall -Wextra -pedantic main.cpp board.cpp movegen.cpp search.cpp eval.cpp tt.cpp -o chess_engine
```

### CMake
```bash
cmake -S . -B build
cmake --build build -j
```

## UCI Commands

Supported:
- `uci`
- `isready`
- `setoption name Hash value <mb>`
- `position startpos [moves ...]`
- `position fen <FEN> [moves ...]`
- `go depth <N>`
- `go movetime <ms>`
- `stop`
- `quit`
- `perft <N>` (full legal-move node count at depth `N`, prints `nodes <count>`)

## Examples

```bash
printf 'uci\nisready\nposition startpos\ngo depth 5\nquit\n' | ./chess_engine
```

```bash
printf 'position startpos\nperft 1\nperft 2\nperft 3\nquit\n' | ./chess_engine
```

## Platform support

Tested on Linux with GCC 11+.

## Notes

Engine emits runtime instrumentation fields in `info` and `perft` output:
- nodes
- time
- nps
- transposition-table hit/probe counters


## Regression checks

```bash
./tests/perft_regression.sh ./chess_engine
./tests/position_regression.sh ./chess_engine
```

## Python training assets (SmallNet / MegaNet)

This repo now includes trainable Python models (no stubs):

- `python/nets.py`
  - `SmallNet`: compact residual policy/value net for faster iteration.
  - `MegaNet`: deeper/wider residual policy/value net for stronger training runs.
- `python/train.py`
  - End-to-end training loop using `.npz` datasets (`planes`, `policy`, `value` arrays).
- `python/opening_book.py`
  - Generates opening books for both models in engine-compatible key format.

Generate books:

```bash
python3 python/opening_book.py --out-dir books
```

Train either net:

```bash
python3 python/train.py --model small --train data/train.npz --valid data/valid.npz --out weights/small.pt
python3 python/train.py --model mega --train data/train.npz --valid data/valid.npz --out weights/mega.pt
```

## Engine dataset generation (binpack with search thinking)

Every `go` search now appends one line to `selfplay.binpack` with:

- position key + board snapshot
- depth / score / nodes
- `value_stm_cp` (label from side-to-move perspective) and `value_white_cp` (white perspective)
- bestmove and PV
- candidate depths
- `thinking` payload (the engine `eval_breakdown` string)

Use `binpackstats` in UCI mode to see how many records are currently stored.
