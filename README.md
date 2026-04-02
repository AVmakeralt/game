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

Default neural weights file names used by the engine:
- NNUE: `NNUE.pt` (auto-falls back to bundled `nn-*.nnue` when `NNUE.pt` is not present)
- Strategy net: `meganet.pt`

Download example strategy weights (`meganet.pt`):
- Google Drive: https://drive.google.com/file/d/1hDyPmXxiTFg6eAU84ya2AqWdLGQCWqBm/view?usp=drive_link

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

## Search algorithms in this engine (including custom ones)

Quick clarification: **UCI is not a search algorithm**. UCI is the communication protocol (`uci`, `position`, `go`, `bestmove`) used by GUIs to talk to the engine.

### 1) Core classical search stack

- **Iterative Deepening**
  - The engine searches depth 1, then 2, then 3... up to target depth/time.
  - This gives a usable best move early and improves move ordering on later depths.

- **Alpha-Beta pruning (negamax form)**
  - Main minimax variant with `alpha`/`beta` bounds to cut branches that cannot affect the final decision.
  - This is the primary Elo-driving search framework.

- **PVS-style zero-window re-search**
  - First move searched with full window, later moves often searched with narrow/null window first.
  - If a narrow search fails high, move is re-searched with a full window.

- **Aspiration windows**
  - Around the previous iteration score, the engine starts with a small window.
  - On fail-low/high it widens the window and re-searches.

- **Quiescence search**
  - At nominal depth 0, engine continues tactical-only exploration (captures/promotions/check-like moves) to reduce horizon artifacts.
  - Uses SEE/delta-style filters and tactical pressure soft-pruning.

- **Null-move pruning / reductions / extensions**
  - Null move can be used to trigger cutoffs in quiet non-check positions.
  - Late reductions and tactical/check extensions are applied to rebalance depth where needed.

- **Transposition table (TT)**
  - Probes/stores with bound types (exact/lower/upper) and depth.
  - Reuses previously-seen positions and improves pruning + move ordering.

### 2) Move-ordering algorithms

- **TT move first** when available.
- **MVV-LVA capture bias** and **promotion bonus**.
- **Killer heuristic** and **history heuristic**.
- **PV hinting** (prefer principal-variation move).
- **Policy and strategy-net priors** injected into ordering.

Good ordering is critical: stronger ordering means more alpha-beta cutoffs and much higher practical depth.

### 3) Custom hybrid algorithms in this repo

- **UGDS root iteration (custom)**
  - Root move selection can switch to a custom uncertainty-guided system.
  - Uses edge fields like belief, uncertainty, tactical risk, proof/refutation style signals, and visit counts.
  - Includes entropy regulation and adversarial-style refutation probes to stabilize risky candidates.

- **M2CTS-style controls (custom)**
  - Optional phase-aware mixing and clade-aware selection controls are exposed in search configuration.
  - Virtual-loss style penalties are used for batch/concurrent selection behavior.

- **Adaptive AB/MCTS switching (custom)**
  - Engine can force classical alpha-beta in tactical or low-material states and allow MCTS features elsewhere.

- **Position-aware time management (custom)**
  - UCI `go` limits are parsed, then per-position tuning adjusts movetime based on tactical status, legal move count, and material/game phase.
  - Final time is capped by remaining clock + increment budget.

- **Neural consistency gate (custom)**
  - Evaluation blends NNUE and strategy-net signals.
  - If small-net vs big-net contribution diverges too much, strategy contribution is damped.
  - This reduces depth-to-depth instability and “evaluation whiplash.”

### 4) Search-adjacent decision layers (not the core tree search)

- **Opening book probe**
  - If a book move exists, it can be played before tree search.

- **Tablebase probing (RAM + optional Syzygy command)**
  - In small-piece endgames, tablebase results can override search and return exact outcomes/moves.

- **Result cache**
  - Reuses previously computed root best moves for repeated keys.

In short: this engine combines a classical alpha-beta backbone with custom uncertainty-guided root logic and neural-aware ordering/evaluation controls.

## Reproducibility spec (exact metrics + thresholds)

This section defines the key metrics and decision gates in code-level terms so behavior is reproducible.

### 1) Uncertainty (used by UGDS)

Static uncertainty estimate:

```text
depthTerm   = 1 / max(1, depth)
swing       = min(1, abs(staticScore - lastIterationScore) / 300)
uncertainty = 0.25 + depthTerm + 0.5 * swing
if transformer_critic_enabled:
    uncertainty += 0.7 * predictedError
uncertainty = clamp(uncertainty, 0.1, 2.5)
```

Interpretation:
- Higher when depth is shallow.
- Higher when the score swings between iterations.
- Optionally raised by transformer-predicted error.

### 2) Tactical risk (used by UGDS edge scoring)

Per-move tactical risk:

```text
risk = 0
risk += 0.35 if move is capture else 0.05
risk += 0.45 if move is promotion
risk += 0.10 if move lands in central squares
risk += clamp(abs(SEE(move)) / 500, 0.0, 0.40)
if strategy_net_enabled:
    risk += 0.3 * clamp(tacticalThreat[0] + tacticalThreat[1], 0, 1)
if transformer_critic_enabled:
    risk += 0.25 * tactical_meta
risk = clamp(risk, 0.0, 1.0)
```

UGDS priority term then uses this risk:

```text
priority = q
         + 70 * sqrt(uncertainty / (1 + visits))
         + 55 * tacticalRisk
         + 35 * prior/(1 + visits)
         - 90 * (refutationStrength * refutationConfidence)
         - (35 if fragile else 0)
```

### 3) Search-method switching (exact gates)

At `go` time:

```text
forceAlphaBeta = adaptiveSwitching && (pieceCount <= 12 || tacticalPosition)
allowMCTS      = features.useMCTS && !forceAlphaBeta
```

So MCTS is explicitly disabled in tactical or low-material states when adaptive switching is enabled.

At root iteration level:

```text
useUGDSIteration = features.useUGDS && (rootMoveCount > 1)
```

If true, root selection uses UGDS iteration; otherwise it stays on the classic ordered-eval root loop.

### 4) Cross-model divergence damping (exact trigger)

When strategy contribution is blended with NNUE:

```text
divergence = abs(nnueContribution - megaContribution)
if divergence > 180: megaContribution /= 3
else if divergence > 120: megaContribution /= 2
```

This is the exact consistency gate currently used to avoid depth-to-depth evaluation whiplash between small and large nets.

### 5) Time-allocation tuning (exact multipliers)

Starting from parsed `movetimeMs`:

```text
if tactical:                * 1.25
if legalCount <= 8:         * 1.50
else if legalCount >= 35:   * 0.85
if pieceCount <= 10:        * 1.40
cap = remainingMs/5 + 2*incrementMs   (if remainingMs > 0)
movetime = max(1, min(tuned, cap))
```

This turns “position-aware timing” into explicit, testable rules.
