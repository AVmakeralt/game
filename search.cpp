#include "search.h"

#include <algorithm>
#include <chrono>
#include <cctype>

namespace search {

namespace {
constexpr int INF = 1000000;
constexpr int MATE = 900000;
int see(const board::Board& b, const movegen::Move& m) {
  auto val = [](char c) {
    switch (std::tolower(static_cast<unsigned char>(c))) {
      case 'p': return 100; case 'n': return 320; case 'b': return 330; case 'r': return 500; case 'q': return 900; default: return 0;
    }
  };
  return val(b.squares[m.to]) - val(b.squares[m.from]);
}
}

Searcher::Searcher(const eval::Params& params, tt::Table* table, bool* stop)
    : params_(params), tt_(table), stop_(stop) {}

std::vector<movegen::Move> Searcher::order(board::Board& b, const std::vector<movegen::Move>& moves, const movegen::Move& ttMove) {
  std::vector<std::pair<int, movegen::Move>> scored;
  scored.reserve(moves.size());
  for (const auto& m : moves) {
    int score = 0;
    if (m == ttMove) score += 1000000;
    if (b.squares[m.to] != '.') score += 500000 + see(b, m);
    if (m.promotion) score += 400000;
    board::Undo u;
    if (b.makeMove(m.from, m.to, m.promotion, u)) {
      if (b.inCheck(b.whiteToMove)) score += 200000;
      b.unmakeMove(m.from, m.to, m.promotion, u);
    }
    scored.push_back({score, m});
  }
  std::sort(scored.begin(), scored.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
  std::vector<movegen::Move> out;
  for (auto& x : scored) out.push_back(x.second);
  return out;
}

int Searcher::quiescence(board::Board& b, int alpha, int beta, int ply) {
  (void)ply;
  ++nodes_;
  int stand = eval::evaluate(b, params_);
  if (stand >= beta) return beta;
  alpha = std::max(alpha, stand);

  auto moves = movegen::generatePseudoLegal(b);
  for (const auto& m : moves) {
    if (b.squares[m.to] == '.' && !m.promotion) continue;
    if (see(b, m) < -120) continue;
    board::Undo u;
    if (!b.makeMove(m.from, m.to, m.promotion, u)) continue;
    int score = -quiescence(b, -beta, -alpha, ply + 1);
    b.unmakeMove(m.from, m.to, m.promotion, u);
    if (score >= beta) return beta;
    alpha = std::max(alpha, score);
  }
  return alpha;
}

int Searcher::alphaBeta(board::Board& b, int depth, int alpha, int beta, int ply, bool allowNull) {
  if (*stop_) return 0;
  if (depth <= 0) return quiescence(b, alpha, beta, ply);
  ++nodes_;

  std::uint64_t key = tt::hash(b);
  tt::Entry tte;
  movegen::Move ttMove{};
  if (tt_ && tt_->probe(key, tte)) {
    ttMove = tte.bestMove;
    if (tte.depth >= depth) {
      if (tte.bound == tt::EXACT) return tte.score;
      if (tte.bound == tt::LOWER && tte.score >= beta) return tte.score;
      if (tte.bound == tt::UPPER && tte.score <= alpha) return tte.score;
    }
  }

  if (allowNull && depth >= 3 && !b.inCheck(b.whiteToMove)) {
    b.whiteToMove = !b.whiteToMove;
    int score = -alphaBeta(b, depth - 1 - 2, -beta, -beta + 1, ply + 1, false);
    b.whiteToMove = !b.whiteToMove;
    if (score >= beta) return beta;
  }

  auto legal = movegen::generateLegal(b);
  if (legal.empty()) return b.inCheck(b.whiteToMove) ? -MATE + ply : 0;

  auto ordered = order(b, legal, ttMove);
  int best = -INF;
  int origAlpha = alpha;
  movegen::Move bestMove{};

  for (size_t i = 0; i < ordered.size(); ++i) {
    const auto& m = ordered[i];
    board::Undo u;
    if (!b.makeMove(m.from, m.to, m.promotion, u)) continue;

    int ext = b.inCheck(b.whiteToMove) ? 1 : 0;
    int newDepth = depth - 1 + ext;
    int reduction = (depth >= 3 && i >= 4 && b.squares[m.to] == '.' && !m.promotion) ? 1 : 0;

    int score;
    if (i == 0) score = -alphaBeta(b, newDepth, -beta, -alpha, ply + 1, true);
    else {
      score = -alphaBeta(b, newDepth - reduction, -alpha - 1, -alpha, ply + 1, true);
      if (score > alpha && score < beta) score = -alphaBeta(b, newDepth, -beta, -alpha, ply + 1, true);
    }
    b.unmakeMove(m.from, m.to, m.promotion, u);

    if (score > best) {
      best = score;
      bestMove = m;
      if (ply == 0) rootBest_ = m;
    }
    alpha = std::max(alpha, score);
    if (alpha >= beta) break;
  }

  if (tt_) {
    tt::Entry e;
    e.key = key;
    e.depth = depth;
    e.score = best;
    e.bestMove = bestMove;
    e.bound = (best <= origAlpha) ? tt::UPPER : (best >= beta ? tt::LOWER : tt::EXACT);
    tt_->store(e);
  }
  return best;
}

Result Searcher::think(board::Board& b, const Limits& l) {
  Result r;
  nodes_ = 0;
  rootBest_ = {};
  int score = 0;
  int alpha = -INF;
  int beta = INF;
  auto start = std::chrono::steady_clock::now();

  for (int d = 1; d <= l.depth && !*stop_; ++d) {
    int window = 20 + d * 10;
    alpha = score - window;
    beta = score + window;
    while (true) {
      score = alphaBeta(b, d, alpha, beta, 0, true);
      if (score <= alpha) {
        alpha -= window;
        window *= 2;
      } else if (score >= beta) {
        beta += window;
        window *= 2;
      } else break;
    }
    r.depth = d;
    r.bestMove = rootBest_;
    r.scoreCp = score;
    auto now = std::chrono::steady_clock::now();
    int elapsed = static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count());
    if (!l.infinite && l.movetimeMs > 0 && elapsed >= l.movetimeMs) break;
  }

  r.nodes = nodes_;
  r.pv = {r.bestMove};
  r.evalBreakdown = eval::breakdown(b, params_);
  r.candidateDepths = {r.depth};
  return r;
}

}  // namespace search
