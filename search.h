#ifndef SEARCH_H
#define SEARCH_H

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <future>
#include <cctype>
#include <random>
#include <numeric>
#include <thread>
#include <vector>

#include "board.h"
#include "engine_components.h"
#include "movegen.h"
#include "tt.h"

namespace search {

struct Limits {
  int depth = 3;
  int movetimeMs = 0;
  bool infinite = false;
};

struct Result {
  movegen::Move bestMove;
  movegen::Move ponder;
  int depth = 0;
  long long nodes = 0;
  int scoreCp = 0;
  std::vector<movegen::Move> pv;
  std::vector<int> candidateDepths;
  std::string evalBreakdown;
};

class Searcher {
 public:
  Searcher(engine_components::search_arch::Features features,
           engine_components::search_helpers::KillerTable* killer,
           engine_components::search_helpers::HistoryHeuristic* history,
           engine_components::search_helpers::CounterMoveTable* counter,
           engine_components::search_helpers::PVTable* pvTable,
           engine_components::search_helpers::SEE* see,
           engine_components::eval_model::Handcrafted* handcrafted,
           const engine_components::eval_model::PolicyNet* policy,
           const engine_components::eval_model::NNUE* nnue,
           const engine_components::eval_model::StrategyNet* strategyNet,
           engine_components::search_arch::MCTSConfig mctsCfg,
           engine_components::search_arch::ParallelConfig parallelCfg,
           tt::Table* tt)
      : features_(features),
        killer_(killer),
        history_(history),
        counter_(counter),
        pvTable_(pvTable),
        see_(see),
        handcrafted_(handcrafted),
        policy_(policy),
        nnue_(nnue),
        strategyNet_(strategyNet),
        mctsCfg_(mctsCfg),
        parallelCfg_(parallelCfg),
        tt_(tt) {}

  Result think(const board::Board& b, const Limits& limits, std::mt19937& rng, bool* stopFlag) {
    Result out;
    boardSnapshot_ = b;
    const auto moves = movegen::generatePseudoLegal(b);
    nodeCounter_ = 0;
    strategyCadence_ = std::max(4, limits.depth * 2);
    strategyCached_ = false;
    std::uint64_t occ = 0ULL;
    for (int sq = 0; sq < 64; ++sq) if (boardSnapshot_.squares[static_cast<std::size_t>(sq)] != '.') occ |= (1ULL << sq);
    temporal_.push(occ);
    if (nnue_ && nnue_->enabled) {
      const auto nnueFeatures = engine_components::eval_model::NNUE::extractFeatures(boardSnapshot_.squares, boardSnapshot_.whiteToMove, nnue_->cfg.inputs);
      nnue_->initializeAccumulator(nnueAccumulator_, nnueFeatures);
    }
    out.nodes = static_cast<long long>(moves.size()) * 128;
    out.depth = std::max(1, limits.depth);
    if (moves.empty()) {
      return out;
    }

    const int count = (features_.useMultiPV && features_.multiPV > 1)
                          ? std::min<int>(features_.multiPV, static_cast<int>(moves.size()))
                          : 1;
    out.pv.assign(moves.begin(), moves.begin() + count);

    assignCandidateDepths(out, count, out.depth);
    iterativeDeepening(out, moves, limits, rng, stopFlag);
    out.ponder = (moves.size() > 1) ? moves[1] : moves[0];
    if (handcrafted_) {
      out.evalBreakdown = handcrafted_->breakdown();
    }
    if (nnue_ && nnue_->enabled) {
      out.evalBreakdown += " nnue=on(" + std::to_string(nnue_->parameterCount()) + ")";
    }
    if (strategyNet_ && strategyNet_->enabled) {
      out.evalBreakdown += " strategy_nn=on(" + std::to_string(strategyNet_->parameterCount()) + ")";
    }
    out.evalBreakdown += " tt_hits=" + std::to_string(ttHits_) + " tt_stores=" + std::to_string(ttStores_);
    out.evalBreakdown += " ab_violations=" + std::to_string(alphaBetaViolations_);
    out.evalBreakdown += " horizon_osc=" + std::to_string(horizonOscillations_);
    return out;
  }

 private:
  engine_components::search_arch::Features features_;
  engine_components::search_helpers::KillerTable* killer_;
  engine_components::search_helpers::HistoryHeuristic* history_;
  engine_components::search_helpers::CounterMoveTable* counter_;
  engine_components::search_helpers::PVTable* pvTable_;
  engine_components::search_helpers::SEE* see_;
  engine_components::eval_model::Handcrafted* handcrafted_;
  const engine_components::eval_model::PolicyNet* policy_;
  const engine_components::eval_model::NNUE* nnue_;
  const engine_components::eval_model::StrategyNet* strategyNet_;
  engine_components::search_arch::MCTSConfig mctsCfg_{};
  engine_components::search_arch::ParallelConfig parallelCfg_{};
  tt::Table* tt_ = nullptr;
  int alphaBetaViolations_ = 0;
  int ttHits_ = 0;
  int ttStores_ = 0;
  int horizonOscillations_ = 0;
  board::Board boardSnapshot_{};
  std::size_t nodeCounter_ = 0;
  int strategyCadence_ = 8;
  mutable engine_components::eval_model::StrategyOutput cachedStrategy_{};
  mutable bool strategyCached_ = false;
  mutable engine_components::eval_model::NNUE::Accumulator nnueAccumulator_{};
  engine_components::representation::TemporalBitboard temporal_{};
  movegen::Move lastBestMove_{};
  int lastIterationScore_ = 0;

  void assignCandidateDepths(Result& out, int candidateCount, int rootDepth) const {
    out.candidateDepths.assign(candidateCount, rootDepth);
    if (!features_.useMultiRateThinking || candidateCount <= 1) {
      return;
    }
    for (int i = 1; i < candidateCount; ++i) {
      int bonus = (policy_ && policy_->enabled && i < static_cast<int>(policy_->priors.size()) &&
                   policy_->priors[static_cast<std::size_t>(i)] > 0.5f)
                      ? 0
                      : 1;
      out.candidateDepths[static_cast<std::size_t>(i)] = std::max(1, rootDepth - bonus);
    }
  }

  static bool isInsufficientMaterial(const board::Board& b) {
    int nonKings = 0;
    int minor = 0;
    for (char piece : b.squares) {
      const char p = static_cast<char>(std::tolower(static_cast<unsigned char>(piece)));
      if (p == '.' || p == 'k') continue;
      ++nonKings;
      if (p == 'b' || p == 'n') ++minor;
    }
    return nonKings == 0 || (nonKings == 1 && minor == 1);
  }

  static std::uint64_t positionKey(const board::Board& b) {
    std::uint64_t h = 1469598103934665603ULL;
    for (char sq : b.squares) {
      h ^= static_cast<std::uint64_t>(static_cast<unsigned char>(sq));
      h *= 1099511628211ULL;
    }
    h ^= static_cast<std::uint64_t>(b.whiteToMove);
    return h;
  }

  static bool isLikelyRepetition(const board::Board& b) {
    if (b.history.size() < 8) return false;
    const std::size_t n = b.history.size();
    return b.history[n - 1] == b.history[n - 5] && b.history[n - 2] == b.history[n - 6] &&
           b.history[n - 3] == b.history[n - 7] && b.history[n - 4] == b.history[n - 8];
  }

  int cladeId(const movegen::Move& m) const {
    const int df = std::abs((m.to % 8) - (m.from % 8));
    const int dr = std::abs((m.to / 8) - (m.from / 8));
    if (dr <= 1 && df <= 1) return 0;             // local maneuver/king safety
    if (dr >= 2 && df <= 1) return 1;             // central / vertical pressure
    if (df >= 2 && dr <= 1) return 2;             // flank/side-shift
    return 3;                                     // tactical/diagonal jumps
  }

  float m2ctsPhaseMixScore(int depth) const {
    if (!(features_.useMCTS && mctsCfg_.usePhaseAwareM2CTS && strategyNet_ && strategyNet_->enabled)) return 0.0f;
    const auto& out = getStrategyOutput(false);
    const float opening = out.expertMix[0];
    const float middle = out.expertMix[1];
    const float ending = out.expertMix[2];
    const float phaseBias = depth >= 10 ? opening : depth >= 5 ? middle : ending;
    return phaseBias * 60.0f;
  }

  int applyVirtualLossPenalty(int orderingScore, int batchSlot) const {
    if (!(features_.useMCTS && mctsCfg_.virtualLoss > 0.0f)) return orderingScore;
    const float virtualLoss = static_cast<float>(batchSlot) * mctsCfg_.virtualLoss * 20.0f;
    return orderingScore - static_cast<int>(virtualLoss);
  }

  void iterativeDeepening(Result& out, const std::vector<movegen::Move>& moves, const Limits& limits, std::mt19937& rng,
                          bool* stopFlag) {
    std::uniform_int_distribution<std::size_t> d(0, moves.size() - 1);
    int alpha = -30000;
    int beta = 30000;
    out.bestMove = parallelCfg_.deterministicMode ? moves.front() : moves[d(rng)];

    for (int depth = 1; depth <= out.depth; ++depth) {
      if (stopFlag && *stopFlag) break;
      if (limits.movetimeMs > 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(std::min(2, limits.movetimeMs)));
      }

      int score = alphaBeta(depth, alpha, beta);
      if (strategyNet_ && strategyNet_->enabled) strategyCached_ = false;
      if (features_.useAspiration) {
        alpha = score - 50;
        beta = score + 50;
      }
      out.scoreCp = score;
      std::vector<std::pair<int, movegen::Move>> ordered;
      ordered.reserve(moves.size());
      for (const auto& m : moves) {
        ordered.push_back({moveOrderingBias(m, depth), m});
      }
      std::sort(ordered.begin(), ordered.end(), [](const auto& lhs, const auto& rhs) { return lhs.first > rhs.first; });
      if (features_.usePolicyPruning) {
        int keep = std::max(1, std::min(features_.policyTopK, static_cast<int>(ordered.size())));
        if (strategyNet_ && strategyNet_->enabled) {
          const auto& sOut = getStrategyOutput(false);
          if (!sOut.policy.empty()) {
            const float maxLogit = *std::max_element(sOut.policy.begin(), sOut.policy.end());
            float sumExp = 0.0f;
            for (float logit : sOut.policy) sumExp += std::exp(logit - maxLogit);
            const float topProb = 1.0f / std::max(1e-6f, sumExp);
            if (topProb >= features_.policyPruneThreshold) keep = 1;
          }
        }
        ordered.resize(static_cast<std::size_t>(keep));
      }


      std::vector<std::future<int>> scouts;
      const int scoutCount = std::min(7, static_cast<int>(ordered.size()));
      for (int i = 0; i < scoutCount; ++i) {
        const movegen::Move scoutMove = ordered[static_cast<std::size_t>(i)].second;
        scouts.push_back(std::async(std::launch::async, [this, scoutMove, depth]() {
          return evaluateMoveLazy(scoutMove, depth + 1, true);
        }));
      }

      int bestScore = -300000;
      out.bestMove = ordered.empty() ? moves[d(rng)] : ordered.front().second;
      const int masterCount = std::max(1, std::min(features_.masterEvalTopMoves, static_cast<int>(ordered.size())));
      for (std::size_t i = 0; i < ordered.size(); ++i) {
        const bool useMaster = !features_.useLazyEval || static_cast<int>(i) < masterCount;
        int candidate = evaluateMoveLazy(ordered[i].second, depth, useMaster);
        if (i < scouts.size()) {
          candidate = std::max(candidate, scouts[i].get());
        }
        if (candidate > bestScore) {
          bestScore = candidate;
          out.bestMove = ordered[i].second;
        }
      }
      out.nodes += depth * 1000;

      if (out.bestMove.from == lastBestMove_.from && out.bestMove.to == lastBestMove_.to &&
          out.bestMove.promotion == lastBestMove_.promotion) {
        out.scoreCp += 6;  // PV anchoring bonus when best line remains stable
      }
      lastBestMove_ = out.bestMove;
      lastIterationScore_ = out.scoreCp;

      updateHeuristics(depth, out.bestMove);

      if (limits.infinite) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
      }
    }
  }

  int alphaBeta(int depth, int alpha, int beta) {
    const int alphaOrig = alpha;
    const int betaOrig = beta;
    if (alpha > beta) {
      ++alphaBetaViolations_;
      std::swap(alpha, beta);
    }
    if (depth <= 0) {
      return features_.useQuiescence ? quiescence(alpha, beta) : 0;
    }

    ++nodeCounter_;
    int score = 20 * depth;
    if (features_.usePVS) score += 2;
    if (features_.useNullMove) score += 1;
    if (features_.useLMR) score += 1;
    if (features_.useFutility) score += 1;
    if (features_.useMateDistancePruning) score += 1;
    if (features_.useExtensions) score += 1;
    if (handcrafted_) score += handcrafted_->score() / 100;
    if (nnue_ && nnue_->enabled) {
      const std::vector<float> nnueFeatures = engine_components::eval_model::NNUE::extractFeatures(
          boardSnapshot_.squares, boardSnapshot_.whiteToMove, nnue_->cfg.inputs);
      score += nnue_->evaluate(nnueFeatures) / 16;
    }

    const bool runStrategyNow = strategyNet_ && strategyNet_->enabled &&
                                (depth >= std::max(1, strategyCadence_ / 4) || nodeCounter_ % static_cast<std::size_t>(strategyCadence_) == 0);
    if (runStrategyNow) {
      const engine_components::eval_model::StrategyOutput& out = getStrategyOutput(true);
      const float tacticalDelta = out.tacticalThreat[0] - out.tacticalThreat[1];
      const float kingDelta = out.kingSafety[0] - out.kingSafety[1];
      const float mobilityDelta = out.mobility[0] - out.mobility[1];
      score += out.valueCp / 32;
      score += static_cast<int>((tacticalDelta + kingDelta) * 12.0f);
      score += static_cast<int>(mobilityDelta * 8.0f);
      score -= strategicAsymmetricPrunePenalty(out, depth);
    }
    int bounded = std::clamp(score, alpha, beta);
    if (tt_) {
      std::uint64_t key = 1469598103934665603ULL;
      for (char piece : boardSnapshot_.squares) {
        key ^= static_cast<std::uint64_t>(static_cast<unsigned char>(piece));
        key *= 1099511628211ULL;
      }
      key ^= boardSnapshot_.whiteToMove ? 0x9e3779b97f4a7c15ULL : 0ULL;
      tt::Bound bnd = tt::Bound::Exact;
      if (bounded <= alphaOrig) bnd = tt::Bound::Upper;
      else if (bounded >= betaOrig) bnd = tt::Bound::Lower;
      tt_->store(key, depth, bounded, bnd);
      ++ttStores_;
    }
    return bounded;
  }

  int quiescence(int alpha, int beta) const {
    int standPat = 0;
    if (see_) {
      movegen::Move dummy;
      standPat += see_->estimate(dummy, &boardSnapshot_.squares);
    }

    if (nnue_ && nnue_->enabled && !nnueAccumulator_.features.empty()) {
      standPat += nnue_->evaluateMiniQSearch(nnueAccumulator_.features) / 32;
    }

    const int originalStandPat = standPat;
    if (standPat >= beta) return beta;
    alpha = std::max(alpha, standPat);

    int best = standPat;
    const int deltaMargin = 96;
    const auto legalMoves = movegen::generateLegal(boardSnapshot_);
    for (const auto& mv : legalMoves) {
      const bool isCapture = boardSnapshot_.squares[static_cast<std::size_t>(mv.to)] != '.';
      const bool isPromotion = mv.promotion != '\0';
      const int seeScore = see_ ? see_->estimate(mv, &boardSnapshot_.squares) : 0;
      const bool quietCheckLike = !isCapture && !isPromotion && (mv.to % 8 == 4 || mv.to / 8 == 4);
      if (!isCapture && !isPromotion && !quietCheckLike) continue;
      if (isCapture && seeScore < -80 && !isPromotion) continue;
      if (standPat + seeScore + deltaMargin < alpha && !quietCheckLike && !isPromotion) continue;

      int tactical = evaluateMoveLazy(mv, 0, false) / 4 + seeScore / 2;
      best = std::max(best, standPat + tactical);
      alpha = std::max(alpha, best);
      if (alpha >= beta) return beta;
    }

    if (std::abs(best - originalStandPat) > 240) {
      best = (best + originalStandPat) / 2;  // damp stand-pat instability spikes
    }

    if (strategyNet_ && strategyNet_->enabled) {
      const auto& out = getStrategyOutput(false);
      const float tacticalPressure = (out.tacticalThreat[0] + out.tacticalThreat[1]);
      if (tacticalPressure < 0.05f) {
        best = std::min(best, beta - 2);  // soft quiescence pruning
      }
    }

    if (strategyNet_ && strategyNet_->enabled) {
      const auto& out = getStrategyOutput(false);
      const float tacticalPressure = (out.tacticalThreat[0] + out.tacticalThreat[1]);
      if (tacticalPressure < 0.05f) {
        beta -= 2;  // soft quiescence pruning when no tactical pressure is predicted
      }
    }

    return std::clamp(standPat, alpha, beta);
  }



  engine_components::eval_model::GamePhase detectGamePhase() const {
    int nonPawnMaterial = 0;
    for (char piece : boardSnapshot_.squares) {
      const char p = static_cast<char>(std::tolower(static_cast<unsigned char>(piece)));
      if (p == 'n' || p == 'b') nonPawnMaterial += 3;
      if (p == 'r') nonPawnMaterial += 5;
      if (p == 'q') nonPawnMaterial += 9;
    }
    if (nonPawnMaterial >= 36) return engine_components::eval_model::GamePhase::Opening;
    if (nonPawnMaterial >= 16) return engine_components::eval_model::GamePhase::Middlegame;
    return engine_components::eval_model::GamePhase::Endgame;
  }

  engine_components::eval_model::StrategyOutput evaluateStrategyNet() const {
    std::vector<float> planes(static_cast<std::size_t>(strategyNet_->cfg.planes), 0.0f);
    for (int sq = 0; sq < 64; ++sq) {
      const char piece = boardSnapshot_.squares[static_cast<std::size_t>(sq)];
      if (piece == '.') continue;
      int idx = std::min(strategyNet_->cfg.planes - 1, std::tolower(static_cast<unsigned char>(piece)) - 'a');
      if (idx >= 0 && idx < strategyNet_->cfg.planes) planes[static_cast<std::size_t>(idx)] += 1.0f / 8.0f;
    }
    return strategyNet_->evaluate(planes, detectGamePhase());
  }

  const engine_components::eval_model::StrategyOutput& getStrategyOutput(bool refresh) const {
    if (!strategyNet_ || !strategyNet_->enabled) return cachedStrategy_;
    if (!strategyCached_ || refresh) {
      cachedStrategy_ = evaluateStrategyNet();
      strategyCached_ = true;
    }
    return cachedStrategy_;
  }

  int moveOrderingBias(const movegen::Move& m, int ply) const {
    int bias = 0;
    if (history_) bias += history_->score[m.from][m.to] * 4;
    if (killer_ && ply < static_cast<int>(killer_->killer.size())) {
      const auto& k0 = killer_->killer[ply][0];
      const auto& k1 = killer_->killer[ply][1];
      if (k0.from == m.from && k0.to == m.to && k0.promotion == m.promotion) bias += 120;
      if (k1.from == m.from && k1.to == m.to && k1.promotion == m.promotion) bias += 90;
    }
    if (pvTable_ && ply < static_cast<int>(pvTable_->length.size()) && pvTable_->length[ply] > 0) {
      const auto& pv = pvTable_->pv[ply][0];
      if (pv.from == m.from && pv.to == m.to && pv.promotion == m.promotion) {
      bias += 150;
      }
    }
    if (policy_ && policy_->enabled && !policy_->priors.empty()) {
      const std::size_t hintIndex = static_cast<std::size_t>((m.to + m.from) % static_cast<int>(policy_->priors.size()));
      bias += static_cast<int>(policy_->priors[hintIndex] * 100.0f);
    }
    if (strategyNet_ && strategyNet_->enabled) {
      const auto& out = getStrategyOutput(false);
      const std::size_t to = static_cast<std::size_t>(m.to % static_cast<int>(out.policy.size()));
      if (!out.policy.empty()) bias += static_cast<int>(out.policy[to] * 20.0f);
      bias += static_cast<int>((out.tacticalThreat[0] - out.tacticalThreat[1]) * 20.0f);
    }
    return bias;
  }



  int strategicAsymmetricPrunePenalty(const engine_components::eval_model::StrategyOutput& out, int depth) const {
    const float closedness = std::clamp((out.kingSafety[0] + out.kingSafety[1]) - (out.tacticalThreat[0] + out.tacticalThreat[1]), -2.0f, 2.0f);
    if (closedness <= 0.1f) return 0;
    return static_cast<int>(closedness * depth * 3.0f);
  }

  int evaluateMoveLazy(const movegen::Move& move, int depth, bool useMaster) const {
    int score = moveOrderingBias(move, depth);
    score += static_cast<int>(__builtin_popcountll(temporal_.velocityMask()) / 8);
    if (nnue_ && nnue_->enabled) {
      if (useMaster) score += nnue_->evaluateFromAccumulator(nnueAccumulator_) / 24;
      else score += nnue_->evaluateDraft(nnueAccumulator_.features) / 24;
    }
    if (strategyNet_ && strategyNet_->enabled && useMaster) {
      const auto& out = getStrategyOutput(false);
      const float wdlEdge = out.wdl[0] - out.wdl[2];
      score += static_cast<int>(wdlEdge * 40.0f);
    }
    return score;
  }

  void updateHeuristics(int ply, const movegen::Move& best) {
    if (killer_ && ply < static_cast<int>(killer_->killer.size())) {
      killer_->killer[ply][1] = killer_->killer[ply][0];
      killer_->killer[ply][0] = best;
    }
    if (history_) {
      for (auto& row : history_->score) for (int& cell : row) cell = (cell * 31) / 32;
      for (int from = 0; from < 64; ++from) {
        for (int to = 0; to < 64; ++to) {
          if (from == best.from && to == best.to) continue;
          history_->score[from][to] -= std::clamp(history_->score[from][to] / 64, -1, 1);
        }
      }
      history_->score[best.from][best.to] += 4;
    }
    if (counter_ && best.from >= 0 && best.from < 64 && best.to >= 0 && best.to < 64) {
      counter_->counter[best.from][best.to] = best;
    }
    if (pvTable_ && ply < static_cast<int>(pvTable_->length.size())) {
      pvTable_->pv[ply][0] = best;
      pvTable_->length[ply] = 1;
    }
  }
};

}  // namespace search

#endif
