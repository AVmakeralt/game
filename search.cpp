#include "search.h"

#include <algorithm>
#include <chrono>
#include <cctype>

namespace search {

namespace {
constexpr int INF = 1000000;
constexpr int MATE = 900000;
constexpr int MAX_PLY = 120;

long long nowMs() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

int pieceValue(char c) {
  switch (std::tolower(static_cast<unsigned char>(c))) {
    case 'p': return 100;
    case 'n': return 320;
    case 'b': return 330;
    case 'r': return 500;
    case 'q': return 900;
    default: return 0;
  }
}

bool isCapture(const board::Board& b, const movegen::Move& m) {
  char moved = b.squares[m.from];
  char dst = b.squares[m.to];
  if (dst != '.') return true;
  return std::tolower(static_cast<unsigned char>(moved)) == 'p' && m.to == b.enPassantSquare;
}

bool isQuiet(const board::Board& b, const movegen::Move& m) {
  return !isCapture(b, m) && !m.promotion;
}

int see(const board::Board& b, const movegen::Move& m) {
  return pieceValue(b.squares[m.to]) - pieceValue(b.squares[m.from]);
}

}  // namespace

Searcher::Searcher(const eval::Params& params, tt::Table* table, bool* stop)
    : params_(params), tt_(table), stop_(stop) {}

int Searcher::historyScore(const movegen::Move& m) const {
  if (m.from < 0 || m.to < 0) return 0;
  return history_[m.from][m.to];
}

std::vector<movegen::Move> Searcher::orderMoves(board::Board& b, const std::vector<movegen::Move>& moves, const movegen::Move& ttMove,
                                                int ply, movegen::Move prevMove) {
  std::vector<std::pair<int, movegen::Move>> scored;
  scored.reserve(moves.size());
  for (const auto& m : moves) {
    int score = 0;
    if (m == ttMove) score += 2'000'000;
    if (m == killers_[ply][0]) score += 900'000;
    else if (m == killers_[ply][1]) score += 850'000;
    if (prevMove.from >= 0 && counterMoves_[prevMove.from][prevMove.to] == m) score += 800'000;

    if (isCapture(b, m)) {
      score += 1'000'000 + see(b, m);
    } else {
      score += historyScore(m);
    }

    board::Undo u;
    if (b.makeMove(m.from, m.to, m.promotion, u)) {
      if (b.inCheck(b.whiteToMove)) score += 300'000;
      b.unmakeMove(m.from, m.to, m.promotion, u);
    }
    scored.push_back({score, m});
  }
  std::sort(scored.begin(), scored.end(), [](const auto& a, const auto& b) { return a.first > b.first; });

  std::vector<movegen::Move> out;
  out.reserve(scored.size());
  for (auto& x : scored) out.push_back(x.second);
  return out;
}

void Searcher::updateHeuristics(const movegen::Move& best, const std::vector<movegen::Move>& tried, int ply, int depth,
                                movegen::Move prevMove) {
  if (best.from < 0) return;
  history_[best.from][best.to] += depth * depth;
  history_[best.from][best.to] = std::clamp(history_[best.from][best.to], -50000, 50000);

  if (best.from >= 0 && best.to >= 0) {
    if (killers_[ply][0] == best) {
    } else {
      killers_[ply][1] = killers_[ply][0];
      killers_[ply][0] = best;
    }
  }

  if (prevMove.from >= 0 && prevMove.to >= 0) counterMoves_[prevMove.from][prevMove.to] = best;

  for (const auto& m : tried) {
    if (m == best) continue;
    if (m.from >= 0 && m.to >= 0) { history_[m.from][m.to] -= depth; history_[m.from][m.to] = std::clamp(history_[m.from][m.to], -50000, 50000); }
  }
}

int Searcher::quiescence(board::Board& b, int alpha, int beta, int ply) {
  ++nodes_;
  if (softNodeStop_ && nodes_ >= softNodeStop_) *stop_ = true;
  if (deadlineMs_ > 0 && (nodes_ & 1023ULL) == 0 && nowMs() >= deadlineMs_) *stop_ = true;
  if (*stop_) return 0;

  int stand = eval::evaluate(b, params_);
  if (stand >= beta) return beta;
  alpha = std::max(alpha, stand);

  auto pseudo = movegen::generatePseudoLegal(b);
  for (const auto& m : pseudo) {
    if (!isCapture(b, m) && !m.promotion) continue;
    if (see(b, m) < -200) continue;

    board::Undo u;
    if (!b.makeMove(m.from, m.to, m.promotion, u)) continue;
    int score = -quiescence(b, -beta, -alpha, ply + 1);
    b.unmakeMove(m.from, m.to, m.promotion, u);

    if (score >= beta) return beta;
    alpha = std::max(alpha, score);
  }
  return alpha;
}

int Searcher::alphaBeta(board::Board& b, int depth, int alpha, int beta, int ply, bool allowNull, movegen::Move prevMove) {
  if (ply >= MAX_PLY) return eval::evaluate(b, params_);
  if (softNodeStop_ && nodes_ >= softNodeStop_) *stop_ = true;
  if (deadlineMs_ > 0 && (nodes_ & 1023ULL) == 0 && nowMs() >= deadlineMs_) *stop_ = true;
  if (*stop_) return 0;

  if (b.halfmoveClock >= 100) return 0;
  std::uint64_t currentKey = tt::hash(b);
  auto itRep = repetition_.find(currentKey);
  if (itRep != repetition_.end() && itRep->second >= 3) return 0;

  bool inCheck = b.inCheck(b.whiteToMove);
  if (depth <= 0) return quiescence(b, alpha, beta, ply);

  ++nodes_;
  int origAlpha = alpha;
  int origBeta = beta;

  alpha = std::max(alpha, -MATE + ply);
  beta = std::min(beta, MATE - ply - 1);
  if (alpha >= beta) return alpha;

  std::uint64_t key = currentKey;
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

  int staticEval = eval::evaluate(b, params_);
  if (!inCheck && depth <= 3) {
    if (staticEval - 120 * depth >= beta) return staticEval;
  }

  if (allowNull && !inCheck && depth >= 3) {
    b.whiteToMove = !b.whiteToMove;
    int R = 2 + depth / 6;
    int score = -alphaBeta(b, depth - 1 - R, -beta, -beta + 1, ply + 1, false, movegen::Move{});
    b.whiteToMove = !b.whiteToMove;
    if (score >= beta) return beta;
  }

  auto legal = movegen::generateLegal(b);
  if (legal.empty()) return inCheck ? -MATE + ply : 0;

  if (ttMove.from < 0 && depth >= 6) {
    (void)alphaBeta(b, depth - 2, alpha, beta, ply, false, prevMove);
    tt::Entry iid;
    if (tt_ && tt_->probe(key, iid)) ttMove = iid.bestMove;
  }

  auto moves = orderMoves(b, legal, ttMove, std::min(ply, 127), prevMove);

  int bestScore = -INF;
  movegen::Move bestMove{};
  std::vector<movegen::Move> tried;
  tried.reserve(moves.size());

  bool pvNode = (beta - alpha > 1);
  for (size_t i = 0; i < moves.size(); ++i) {
    const auto& m = moves[i];
    bool quiet = isQuiet(b, m);
    bool capture = isCapture(b, m);

    if (depth <= 4 && i >= static_cast<size_t>(6 + depth * 2) && !inCheck && quiet) {
      continue;
    }

    if (!inCheck && depth <= 3 && quiet && staticEval + 80 * depth <= alpha) {
      continue;
    }

    board::Undo u;
    if (!b.makeMove(m.from, m.to, m.promotion, u)) continue;

    std::uint64_t childKey = tt::hash(b);
    ++repetition_[childKey];

    int ext = b.inCheck(b.whiteToMove) ? 1 : 0;
    int lmr = 0;
    if (depth >= 3 && i >= 3 && !inCheck && quiet) {
      lmr = 1 + (depth >= 6 && i >= 8 ? 1 : 0);
      if (capture) lmr = std::max(0, lmr - 1);
    }

    int newDepth = std::max(0, depth - 1 + ext - lmr);

    int score;
    if (i == 0) {
      score = -alphaBeta(b, newDepth, -beta, -alpha, ply + 1, true, m);
    } else {
      score = -alphaBeta(b, newDepth, -alpha - 1, -alpha, ply + 1, true, m);
      if (score > alpha && (pvNode || score < beta)) {
        score = -alphaBeta(b, depth - 1 + ext, -beta, -alpha, ply + 1, true, m);
      }
    }

    b.unmakeMove(m.from, m.to, m.promotion, u);
    auto itChild = repetition_.find(childKey);
    if (itChild != repetition_.end()) {
      if (--itChild->second == 0) repetition_.erase(itChild);
    }
    tried.push_back(m);

    if (score > bestScore) {
      bestScore = score;
      bestMove = m;
      if (ply == 0) rootBest_ = m;
    }
    alpha = std::max(alpha, score);

    if (alpha >= beta) {
      updateHeuristics(bestMove, tried, std::min(ply, 127), depth, prevMove);
      break;
    }
  }

  if (tt_) {
    tt::Entry e;
    e.key = key;
    e.depth = depth;
    e.score = bestScore;
    e.bestMove = bestMove;
    if (bestScore <= origAlpha) e.bound = tt::UPPER;
    else if (bestScore >= origBeta) e.bound = tt::LOWER;
    else e.bound = tt::EXACT;
    tt_->store(e);
  }

  return bestScore;
}

Result Searcher::think(board::Board& b, const Limits& l) {
  Result r;
  nodes_ = 0;
  rootBest_ = {};
  rootPonder_ = {};
  *stop_ = false;
  repetition_.clear();
  repetition_[tt::hash(b)] = 1;

  if (l.movetimeMs > 0) {
    softNodeStop_ = static_cast<std::uint64_t>(std::max(10000, l.movetimeMs * 2000));
    deadlineMs_ = nowMs() + l.movetimeMs;
  } else {
    softNodeStop_ = 0;
    deadlineMs_ = 0;
  }

  int score = 0;
  auto start = std::chrono::steady_clock::now();

  for (int depth = 1; depth <= std::max(1, l.depth) && !*stop_; ++depth) {
    int delta = 18 + depth * 8;
    int alpha = std::max(-INF, score - delta);
    int beta = std::min(INF, score + delta);

    while (!*stop_) {
      score = alphaBeta(b, depth, alpha, beta, 0, true, movegen::Move{});
      if (score <= alpha) {
        alpha = std::max(-INF, alpha - delta);
        delta *= 2;
      } else if (score >= beta) {
        beta = std::min(INF, beta + delta);
        delta *= 2;
      } else {
        break;
      }
    }

    r.depth = depth;
    r.bestMove = rootBest_;
    r.scoreCp = score;
    r.nodes = nodes_;

    if (r.bestMove.from >= 0) {
      board::Undo u;
      if (b.makeMove(r.bestMove.from, r.bestMove.to, r.bestMove.promotion, u)) {
        auto pm = movegen::generateLegal(b);
        if (!pm.empty()) rootPonder_ = pm.front();
        b.unmakeMove(r.bestMove.from, r.bestMove.to, r.bestMove.promotion, u);
      }
    }

    auto now = std::chrono::steady_clock::now();
    int elapsed = static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count());
    if (!l.infinite && l.movetimeMs > 0 && elapsed >= l.movetimeMs) break;
  }

  r.ponder = rootPonder_;
  r.pv = {r.bestMove};
  r.evalBreakdown = eval::breakdown(b, params_);
  r.candidateDepths = {r.depth};
  return r;
}

}  // namespace search
