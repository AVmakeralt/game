# Engine Roadmap / Architecture ("#2 best engine" plan)

This commit adds first-class scaffolding for all major engine domains requested.
Implementation status is intentionally **scaffolded** for many items so they can be filled incrementally without changing public wiring.

## Core search & tree
- [x] Alpha-beta search entry and score loop (`search::Searcher::alphaBeta`).
- [x] PVS / aspiration / quiescence feature switches.
- [x] Iterative deepening loop.
- [x] Move ordering scaffolding (killer, history, counter/refutation).
- [x] Null-move / LMR / futility / mate-distance / extension toggles.
- [x] Multi-PV plumbing.
- [x] MCTS configuration hook.
- [x] Parallel/asynchronous configuration hooks.

## Data structures & representation
- [x] 64-bit bitboard type.
- [x] 128-bit bitboard type.
- [x] Magic/attack table scaffolding.
- [x] Zobrist hashing scaffold.
- [x] PV table and killer table structures.
- [x] Repetition + fifty-move tracking scaffold.
- [x] Incremental eval integration point.

## Evaluation (handcrafted)
- [x] Material/PSQT/pawn structure/king safety/mobility/space terms.
- [x] Bishop pair/minor dynamics.
- [x] Rook file activity and seventh rank bucket.
- [x] King/pawn tropism.
- [x] Tempo/initiative/time-awareness terms.

## Evaluation (learned)
- [x] NNUE configuration and loading hook.
- [x] Policy prior container.
- [x] Hybrid eval extension point.
- [x] Training infra toggles (self-play/supervised/distillation/replay buffer).

## Move selection & pruning helpers
- [x] SEE placeholder.
- [x] PVS + aspiration integration points.
- [x] IID/LMP/probabilistic pruning represented by feature toggles.
- [x] Tactical solver integration slot (via helper namespace entry points).

## Opening knowledge
- [x] Opening book model (polyglot/custom hook).
- [x] Book prep module with novelty/pruning flags.
- [x] Online/book-update integration point via book path & runtime options.

## Time management / practical play
- [x] Time manager with remaining/increment and per-move allocation.
- [x] Game mode + pondering flags.
- [x] Adaptive style parameters supported through runtime option and harness maps.

## Databases / formats / tooling
- [x] PGN/EPD feature toggles.
- [x] Benchmark/perft command hooks.
- [x] UCI integration hooks for options and analysis output.

## Testing / tuning / evaluation
- [x] Regression/Elo/self-play harness switches.
- [x] Parameter dictionary for optimization/tuning.
- [x] Confidence/match framework integration points.

