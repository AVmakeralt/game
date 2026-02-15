# Competitive Engine Strategy

## References to Study
- Stockfish: alpha-beta + tuning + NNUE hybrid benchmark.
- AlphaZero: full policy/value + MCTS reinforcement learning template.
- Leela Chess Zero: open community implementation of AlphaZero-style training.

## Big-picture recommendation
Start with an alpha-beta + NNUE hybrid backbone, then run parallel research for larger policy/value nets and MCTS.
This gives practical CPU strength now and preserves a path to frontier neural ideas later.

## Delivery phases
- Phase 0: movegen/perft correctness, tests, profiling baselines.
- Phase 1: world-class search stack and scalable threading.
- Phase 2: elite handcrafted eval + incremental NNUE pipeline.
- Phase 3: self-play RL, large-net experiments, policy priors, distillation.
- Phase 4: continuous Elo gating, CI matches, low-level optimization and tournaments.
