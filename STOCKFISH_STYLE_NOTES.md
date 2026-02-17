# Stockfish-Style Patterns Adopted (C++, no neural nets)

Implemented search stack now includes:

1. Adaptive move ordering pipeline:
   - PV/TT first, winning captures (SEE), promotions/forcing scans, killer/counter, adaptive history.
   - Dynamic PV weighting from per-game successful lines.
2. Dynamic LMR:
   - Reduction depends on move order, threat density, king safety, and material imbalance.
3. Dynamic selective extensions:
   - Capture/recapture pressure, promotion/passed-pawn style pushes, and threat-score extension.
4. Enhanced quiescence:
   - Forcing-only tactical set with SEE-based capture pruning and variable q-depth by threat context.
5. Multi-tier transposition table:
   - PV/tactical/quiet buckets with depth+priority replacement and age handling.
6. Endgame heuristics (tablebase-free hooks):
   - Opposition, triangulation, and zugzwang fields integrated in endgame scoring path.
7. Incremental-style heuristic updates:
   - History success/fail accounting and adaptive history weighting.
8. Selective deepening bias:
   - Multi-rate candidate depths with tactical hotspot preference.
9. Opening novelty + dynamic line weighting:
   - Novelty tracking and per-line weight updates during search output.
10. Parallelization preparation:
   - Thread/NUMA flags retained in architecture for root/thread-local rollout.
11. Adaptive aspiration windows:
   - Window scaling with depth, phase, and evaluation volatility.

All components are implemented in C++ and kept CPU-first.
