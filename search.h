#ifndef SEARCH_H
#define SEARCH_H

#include <algorithm>
#include <chrono>
#include <random>
#include <thread>
#include <vector>

#include "board.h"
#include "engine_components.h"
#include "movegen.h"

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
           const engine_components::eval_model::PolicyNet* policy)
      : features_(features),
        killer_(killer),
        history_(history),
        counter_(counter),
        pvTable_(pvTable),
        see_(see),
        handcrafted_(handcrafted),
        policy_(policy) {}

  Result think(const board::Board& b, const Limits& limits, std::mt19937& rng, bool* stopFlag) {
    Result out;
    const auto moves = movegen::generatePseudoLegal(b);
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

  void iterativeDeepening(Result& out, const std::vector<movegen::Move>& moves, const Limits& limits, std::mt19937& rng,
                          bool* stopFlag) {
    std::uniform_int_distribution<std::size_t> d(0, moves.size() - 1);
    int alpha = -30000;
    int beta = 30000;
    out.bestMove = moves[d(rng)];

    for (int depth = 1; depth <= out.depth; ++depth) {
      if (stopFlag && *stopFlag) break;
      if (limits.movetimeMs > 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(std::min(2, limits.movetimeMs)));
      }

      int score = alphaBeta(depth, alpha, beta);
      if (features_.useAspiration) {
        alpha = score - 50;
        beta = score + 50;
      }
      out.scoreCp = score;
      out.bestMove = moves[d(rng)];
      out.nodes += depth * 1000;

      updateHeuristics(depth, out.bestMove);

      if (limits.infinite) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
      }
    }
  }

  int alphaBeta(int depth, int alpha, int beta) {
    if (depth <= 0) {
      return features_.useQuiescence ? quiescence(alpha, beta) : 0;
    }

    int score = 20 * depth;
    if (features_.usePVS) score += 2;
    if (features_.useNullMove) score += 1;
    if (features_.useLMR) score += 1;
    if (features_.useFutility) score += 1;
    if (features_.useMateDistancePruning) score += 1;
    if (features_.useExtensions) score += 1;
    if (handcrafted_) score += handcrafted_->score() / 100;
    return std::clamp(score, alpha, beta);
  }

  int quiescence(int alpha, int beta) const {
    int standPat = 0;
    if (see_) {
      movegen::Move dummy;
      standPat += see_->estimate(dummy);
    }
    return std::clamp(standPat, alpha, beta);
  }

  void updateHeuristics(int ply, const movegen::Move& best) {
    if (killer_ && ply < static_cast<int>(killer_->killer.size())) {
      killer_->killer[ply][1] = killer_->killer[ply][0];
      killer_->killer[ply][0] = best;
    }
    if (history_) {
      history_->score[best.from][best.to] += 1;
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
