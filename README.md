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
- `perft <N>`

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
