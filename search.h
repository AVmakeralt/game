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
           const engine_components::eval_model::TransformerCritic* transformerCritic,
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
        transformerCritic_(transformerCritic),
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
      out.evalBreakdown += " meganet_lc0=on(" + std::to_string(strategyNet_->parameterCount()) + ")";
    }
    if (policy_ && policy_->enabled) {
      out.evalBreakdown += " blitz_net=on(" + std::to_string(policy_->parameterCount()) + ")";
    }
    if (transformerCritic_ && transformerCritic_->enabled) {
      out.evalBreakdown += " transformer=on(" + std::to_string(transformerCritic_->parameterCount()) + ")";
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
  const engine_components::eval_model::TransformerCritic* transformerCritic_;
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
  struct UGDSEdge {
    movegen::Move move{};
    float q = 0.0f;
    float proof = 0.0f;
    float belief = 0.0f;
    float uncertainty = 1.0f;
    float tacticalRisk = 0.0f;
    float prior = 0.0f;
    int visits = 0;
    int lowerBound = -300000;
    int upperBound = 300000;
  };

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
      int bestScore = -300000;
      if (features_.useUGDS) {
        bestScore = runUGDSRootIteration(moves, depth, out.bestMove, rng);
      } else {
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
      }
      out.scoreCp = bestScore;
      out.nodes += depth * 1000 + static_cast<long long>(moves.size()) * std::max(1, depth) * 8;

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

  float staticUncertaintyEstimate(int depth, int staticScore) const {
    const float depthTerm = 1.0f / std::max(1, depth);
    const float swing = std::min(1.0f, std::abs(staticScore - lastIterationScore_) / 300.0f);
    float uncertainty = 0.25f + depthTerm + swing * 0.5f;
    if (transformerCritic_ && transformerCritic_->enabled) {
      const auto meta = transformerCritic_->evaluate(boardSnapshot_.squares, boardSnapshot_.whiteToMove, depth, staticScore);
      uncertainty += meta.predictedError * 0.7f;
    }
    return std::clamp(uncertainty, 0.1f, 2.5f);
  }

  float tacticalRiskEstimate(const movegen::Move& move) const {
    const bool capture = boardSnapshot_.squares[static_cast<std::size_t>(move.to)] != '.';
    const bool promotion = move.promotion != '\0';
    const bool central = (move.to % 8 >= 2 && move.to % 8 <= 5 && move.to / 8 >= 2 && move.to / 8 <= 5);
    const int seeScore = see_ ? see_->estimate(move, &boardSnapshot_.squares) : 0;
    float risk = 0.0f;
    risk += capture ? 0.35f : 0.05f;
    risk += promotion ? 0.45f : 0.0f;
    risk += central ? 0.10f : 0.0f;
    risk += std::clamp(std::abs(seeScore) / 500.0f, 0.0f, 0.40f);
    if (strategyNet_ && strategyNet_->enabled) {
      const auto& out = getStrategyOutput(false);
      risk += std::clamp(out.tacticalThreat[0] + out.tacticalThreat[1], 0.0f, 1.0f) * 0.3f;
    }
    if (transformerCritic_ && transformerCritic_->enabled) {
      const auto meta = transformerCritic_->evaluate(boardSnapshot_.squares, boardSnapshot_.whiteToMove, 4, 0);
      risk += meta.tactical * 0.25f;
    }
    return std::clamp(risk, 0.0f, 1.0f);
  }

  float ugdsPriority(const UGDSEdge& edge) const {
    constexpr float A = 70.0f;
    constexpr float B = 55.0f;
    constexpr float C = 35.0f;
    const float uncertaintyBonus = std::sqrt(edge.uncertainty / static_cast<float>(1 + edge.visits));
    const float priorBonus = edge.prior / static_cast<float>(1 + edge.visits);
    return edge.q + A * uncertaintyBonus + B * edge.tacticalRisk + C * priorBonus;
  }

  int searchUGDSChild(const movegen::Move& move, int depth, float tacticalRisk, float* outBelief, float* outUncertainty) {
    board::Undo undo;
    if (!boardSnapshot_.makeMove(move.from, move.to, move.promotion, undo)) {
      if (outBelief) *outBelief = -30000.0f;
      if (outUncertainty) *outUncertainty = 2.5f;
      return -30000;
    }

    std::uint64_t occ = 0ULL;
    for (int sq = 0; sq < 64; ++sq) if (boardSnapshot_.squares[static_cast<std::size_t>(sq)] != '.') occ |= (1ULL << sq);
    temporal_.push(occ);

    const int beliefScore = evaluateMoveLazy(move, depth, true);
    const int burstExtension = tacticalRisk > 0.65f ? 2 : 0;
    const int proofDepth = std::max(1, depth - 1 + burstExtension);
    const int proof = -alphaBeta(proofDepth, -30000, 30000);
    boardSnapshot_.unmakeMove(move.from, move.to, move.promotion, undo);

    const int mixed = static_cast<int>(proof * 0.75f + beliefScore * 0.25f);
    if (outBelief) *outBelief = static_cast<float>(beliefScore);
    if (outUncertainty) {
      const float disagreement = std::min(2.0f, std::abs(proof - beliefScore) / 250.0f);
      *outUncertainty = std::clamp(0.2f + disagreement, 0.1f, 2.5f);
    }
    return mixed;
  }

  int runUGDSRootIteration(const std::vector<movegen::Move>& rootMoves, int depth, movegen::Move& bestMove, std::mt19937& rng) {
    std::vector<UGDSEdge> edges;
    edges.reserve(rootMoves.size());
    float priorSum = 0.0f;
    for (const auto& mv : rootMoves) {
      UGDSEdge edge;
      edge.move = mv;
      edge.belief = static_cast<float>(evaluateMoveLazy(mv, depth, false));
      edge.proof = edge.belief;
      edge.q = edge.belief;
      edge.tacticalRisk = tacticalRiskEstimate(mv);
      edge.uncertainty = staticUncertaintyEstimate(depth, static_cast<int>(edge.belief));
      edge.prior = std::max(0.01f, static_cast<float>(moveOrderingBias(mv, depth) + 200.0f));
      priorSum += edge.prior;
      edges.push_back(edge);
    }
    for (auto& edge : edges) edge.prior /= std::max(1e-3f, priorSum);

    const int iterations = std::max(static_cast<int>(edges.size()) * 3, std::min(mctsCfg_.miniBatchSize / 8, 128));
    std::uniform_int_distribution<std::size_t> edgeDist(0, edges.size() - 1);

    for (int i = 0; i < iterations; ++i) {
      std::size_t pick = 0;
      float bestPriority = -1e9f;
      if (i < static_cast<int>(edges.size())) {
        pick = static_cast<std::size_t>(i);
      } else {
        for (std::size_t idx = 0; idx < edges.size(); ++idx) {
          const float p = ugdsPriority(edges[idx]);
          if (p > bestPriority) {
            bestPriority = p;
            pick = idx;
          }
        }
        if (edges[pick].uncertainty > 1.2f && (i % 5 == 0)) {
          pick = edgeDist(rng);
        }
      }

      auto& edge = edges[pick];
      float belief = edge.belief;
      float freshUncertainty = edge.uncertainty;
      const int childScore = searchUGDSChild(edge.move, depth, edge.tacticalRisk, &belief, &freshUncertainty);
      edge.visits += 1;
      edge.proof = std::max(edge.proof, static_cast<float>(childScore));
      edge.belief = (edge.belief * static_cast<float>(edge.visits - 1) + belief) / static_cast<float>(edge.visits);
      edge.q = 0.70f * edge.proof + 0.30f * edge.belief;
      edge.lowerBound = std::max(edge.lowerBound, childScore - 16);
      edge.upperBound = std::min(edge.upperBound, childScore + 16);
      edge.uncertainty = std::max(0.1f, (edge.uncertainty * 0.8f + freshUncertainty * 0.2f) * 0.92f);
      edge.tacticalRisk = std::clamp(edge.tacticalRisk * 0.85f + tacticalRiskEstimate(edge.move) * 0.15f, 0.0f, 1.0f);
    }

    const auto bestIt = std::max_element(edges.begin(), edges.end(), [](const UGDSEdge& lhs, const UGDSEdge& rhs) {
      if (lhs.q == rhs.q) return lhs.visits < rhs.visits;
      return lhs.q < rhs.q;
    });
    if (bestIt == edges.end()) {
      bestMove = rootMoves.front();
      return 0;
    }
    bestMove = bestIt->move;
    return static_cast<int>(bestIt->q);
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
    float nnueWeight = 1.0f;
    float megaWeight = 1.0f;
    float blitzWeight = 1.0f;
    if (transformerCritic_ && transformerCritic_->enabled) {
      const auto meta = transformerCritic_->evaluate(boardSnapshot_.squares, boardSnapshot_.whiteToMove, depth, lastIterationScore_);
      nnueWeight = meta.netWeights[0];
      megaWeight = meta.netWeights[1];
      blitzWeight = meta.netWeights[2];
    }
    if (nnue_ && nnue_->enabled) {
      if (useMaster) score += static_cast<int>(nnueWeight * (nnue_->evaluateFromAccumulator(nnueAccumulator_) / 24.0f));
      else score += static_cast<int>(nnueWeight * (nnue_->evaluateDraft(nnueAccumulator_.features) / 24.0f));
    }
    if (strategyNet_ && strategyNet_->enabled && useMaster) {
      const auto& out = getStrategyOutput(false);
      const float wdlEdge = out.wdl[0] - out.wdl[2];
      score += static_cast<int>(megaWeight * wdlEdge * 40.0f);
    }
    if (policy_ && policy_->enabled && !policy_->priors.empty()) {
      const std::size_t idx = static_cast<std::size_t>((move.from * 13 + move.to * 7 + depth) % static_cast<int>(policy_->priors.size()));
      score += static_cast<int>(blitzWeight * policy_->priors[idx] * 60.0f);
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
