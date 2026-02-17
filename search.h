#ifndef SEARCH_H
#define SEARCH_H

#include <array>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "board.h"
#include "eval.h"
#include "movegen.h"
#include "tt.h"

namespace search {

struct Limits {
  int depth = 5;
  int movetimeMs = 0;
  bool infinite = false;
};

struct Result {
  movegen::Move bestMove;
  movegen::Move ponder;
  int depth = 0;
  std::uint64_t nodes = 0;
  int scoreCp = 0;
  std::vector<movegen::Move> pv;
  std::string evalBreakdown;
  std::vector<int> candidateDepths;
};

class Searcher {
 public:
  Searcher(const eval::Params& params, tt::Table* table, bool* stop);
  Result think(board::Board& b, const Limits& l);

 private:
  int alphaBeta(board::Board& b, int depth, int alpha, int beta, int ply, bool allowNull, movegen::Move prevMove);
  int quiescence(board::Board& b, int alpha, int beta, int ply);
  std::vector<movegen::Move> orderMoves(board::Board& b, const std::vector<movegen::Move>& moves, const movegen::Move& ttMove,
                                        int ply, movegen::Move prevMove);
  int historyScore(const movegen::Move& m) const;
  void updateHeuristics(const movegen::Move& best, const std::vector<movegen::Move>& tried, int ply, int depth, movegen::Move prevMove);

  const eval::Params& params_;
  tt::Table* tt_;
  bool* stop_;
  std::uint64_t nodes_ = 0;
  movegen::Move rootBest_{};
  movegen::Move rootPonder_{};
  std::array<std::array<int, 64>, 64> history_{};
  std::array<std::array<movegen::Move, 2>, 128> killers_{};
  std::array<std::array<movegen::Move, 64>, 64> counterMoves_{};
  std::uint64_t softNodeStop_ = 0;
  long long deadlineMs_ = 0;
  std::unordered_map<std::uint64_t, int> repetition_;
};

}  // namespace search

#endif
