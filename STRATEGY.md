# Competitive Engine Strategy (Pure C++)

## Direction
Build a no-NN, CPU-first alpha-beta engine with Stockfish-style search engineering and stronger adaptive heuristics.

## Priority implementation order
1. Move generation correctness + perft + legality regression tests.
2. Move ordering dominance (PV/captures/forcing/killer/counter/adaptive history).
3. Pruning and reductions (adaptive LMR, phase-aware futility, selective deepening).
4. Quiescence quality and SEE pruning in tactical zones.
5. Multi-tier TT and cache quality under long matches.
6. Endgame heuristic expansion (opposition, triangulation, zugzwang, draw patterns).
7. Parallel scaling (root split, thread-local heuristics, NUMA memory layout).
8. Continuous Elo gating and performance profiling.

## Benchmarking
- Perft suites for correctness.
- Tactical regression suite for q-search and pruning safety.
- Long match Elo testing for each patch.
