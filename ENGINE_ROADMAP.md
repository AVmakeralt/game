# Engine Roadmap / Architecture (Pure C++, no NN)

This roadmap keeps all major non-NN engine domains requested and implements/retains them as active code paths or integration hooks.

## Core search & tree
- [x] Alpha-beta/PVS search backbone.
- [x] Iterative deepening and adaptive aspiration windows.
- [x] Quiescence search with forcing-move focus and SEE pruning.
- [x] Adaptive move ordering (PV/TT, captures, killers, history, counters, tactical prefilter).
- [x] Null-move, adaptive LMR, phase-aware futility, mate-distance pruning.
- [x] Selective extensions + selective deepening bias.
- [x] Multi-PV plumbing and per-candidate depth assignment.
- [x] MCTS hook (disabled by default, non-NN placeholder).
- [x] Parallel/asynchronous/NUMA configuration hooks.

## Data structures & representation
- [x] 64-bit and 128-bit bitboard types.
- [x] Magic/attack table scaffolding.
- [x] Zobrist hash scaffolding + repetition/fifty-move tracker.
- [x] PV table, killer table, history and counter/refutation tables.
- [x] Multi-tier transposition table (PV/tactical/quiet) with adaptive replacement.
- [x] Incremental heuristic update hooks.

## Evaluation (handcrafted)
- [x] Material/PSQT/pawn structure/king safety/mobility/space terms.
- [x] Bishop pair/minor dynamics.
- [x] Rook activity terms.
- [x] King/pawn tropism.
- [x] Tempo/initiative/time-awareness terms.
- [x] Endgame heuristics (opposition/triangulation/zugzwang hooks).

## Move selection & pruning helpers
- [x] SEE helper.
- [x] IID/LMP/probabilistic pruning toggles + usage.
- [x] Tactical solver slot for fast forcing-score scans.

## Opening knowledge
- [x] Opening book model (polyglot/custom hook).
- [x] Novelty detection + dynamic opening line weights.
- [x] Opening result cache persistence across sessions.

## Time management / practical play
- [x] Time manager with remaining/increment allocation.
- [x] Game mode and pondering fields.
- [x] Adaptive style/tuning parameter container.

## Databases / formats / tooling
- [x] PGN/EPD toggles.
- [x] Benchmark/perft command hooks.
- [x] UCI integration hooks and diagnostics.

## Testing / tuning / evaluation
- [x] Regression/Elo/self-play harness switches.
- [x] Parameter dictionary for tuning.
- [x] Confidence/match framework hooks.
