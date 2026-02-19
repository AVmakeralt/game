#include "eval.h"

#include <array>
#include <cctype>
#include <cmath>
#include <sstream>

namespace eval {

void initialize(Params& params) {
  if (params.piece[0] <= 0) params.piece[0] = 100;
  if (params.piece[1] <= 0) params.piece[1] = 320;
  if (params.piece[2] <= 0) params.piece[2] = 330;
  if (params.piece[3] <= 0) params.piece[3] = 500;
  if (params.piece[4] <= 0) params.piece[4] = 900;
}

static int knightPst(int sq) {
  static const std::array<int, 64> knight = {
      -50, -40, -30, -30, -30, -30, -40, -50, -40, -20, 0,   5,   5,   0,   -20, -40,
      -30, 5,   10,  15,  15,  10,  5,   -30, -30, 0,   15,  20,  20,  15,  0,   -30,
      -30, 5,   15,  20,  20,  15,  5,   -30, -30, 0,   10,  15,  15,  10,  0,   -30,
      -40, -20, 0,   0,   0,   0,   -20, -40, -50, -40, -30, -30, -30, -30, -40, -50};
  return knight[sq];
}

static int pst(char piece, int sq) {
  char p = static_cast<char>(std::tolower(static_cast<unsigned char>(piece)));
  if (p == 'n') return knightPst(sq);
  return 0;
}

int evaluate(const board::Board& b, const Params& params) {
  int score = 0;
  int whiteBishops = 0;
  int blackBishops = 0;
  int whiteRooks = 0;
  int blackRooks = 0;
  int whiteMinor = 0;
  int blackMinor = 0;
  int whiteMajor = 0;
  int blackMajor = 0;
  int whiteKingSq = -1;
  int blackKingSq = -1;
  std::array<int, 8> whitePawnsByFile{};
  std::array<int, 8> blackPawnsByFile{};

  for (int sq = 0; sq < 64; ++sq) {
    const char c = b.squares[sq];
    if (c == '.') continue;

    const bool white = std::isupper(static_cast<unsigned char>(c));
    const char p = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    const int idx = p == 'p' ? 0 : p == 'n' ? 1 : p == 'b' ? 2 : p == 'r' ? 3 : p == 'q' ? 4 : 5;
    const int material = params.piece[idx];
    const int psq = pst(c, white ? sq : (56 ^ sq));
    const int term = material + psq;

    if (white) {
      score += term;
    } else {
      score -= term;
    }

    if (c == 'B') ++whiteBishops;
    if (c == 'b') ++blackBishops;
    if (c == 'R') ++whiteRooks;
    if (c == 'r') ++blackRooks;
    if (c == 'N' || c == 'B') ++whiteMinor;
    if (c == 'n' || c == 'b') ++blackMinor;
    if (c == 'R' || c == 'Q') ++whiteMajor;
    if (c == 'r' || c == 'q') ++blackMajor;
    if (c == 'K') whiteKingSq = sq;
    if (c == 'k') blackKingSq = sq;
    if (c == 'P') ++whitePawnsByFile[static_cast<std::size_t>(sq % 8)];
    if (c == 'p') ++blackPawnsByFile[static_cast<std::size_t>(sq % 8)];
  }

  if (whiteBishops >= 2) score += params.bishopPairBonus;
  if (blackBishops >= 2) score -= params.bishopPairBonus;
  if (whiteRooks >= 2) score += params.rookPairBonus;
  if (blackRooks >= 2) score -= params.rookPairBonus;

  score += (whiteMinor - whiteMajor) * params.minorVsMajorImbalance;
  score -= (blackMinor - blackMajor) * params.minorVsMajorImbalance;

  auto pawnStructurePenalty = [&](bool whiteSide) {
    int penalty = 0;
    const auto& pawnsByFile = whiteSide ? whitePawnsByFile : blackPawnsByFile;
    for (int file = 0; file < 8; ++file) {
      int count = pawnsByFile[static_cast<std::size_t>(file)];
      if (count <= 0) continue;

      if (count > 1) {
        penalty += (count - 1) * params.doubledPawnPenalty;
      }

      const bool hasLeft = file > 0 && pawnsByFile[static_cast<std::size_t>(file - 1)] > 0;
      const bool hasRight = file < 7 && pawnsByFile[static_cast<std::size_t>(file + 1)] > 0;
      if (!hasLeft && !hasRight) {
        penalty += count * params.isolatedPawnPenalty;
      }

      if ((file == 2 || file == 3 || file == 4 || file == 5) && !hasLeft && !hasRight) {
        penalty += count * params.backwardPawnPenalty;
      }
    }
    return penalty;
  };

  score -= pawnStructurePenalty(true);
  score += pawnStructurePenalty(false);

  auto kingSafetyMask = [&](int kingSq, bool whiteSide) {
    if (kingSq < 0) return 0;
    const int rank = kingSq / 8;
    const int backRank = whiteSide ? 0 : 7;
    const int centerDistance = std::abs((kingSq % 8) - 3) + std::abs(rank - 3);
    const int shieldRank = whiteSide ? rank + 1 : rank - 1;
    int shield = 0;
    for (int df = -1; df <= 1; ++df) {
      const int file = (kingSq % 8) + df;
      if (file < 0 || file > 7 || shieldRank < 0 || shieldRank > 7) continue;
      const int sq = shieldRank * 8 + file;
      const char pawn = whiteSide ? 'P' : 'p';
      if (b.squares[static_cast<std::size_t>(sq)] == pawn) ++shield;
    }
    const int openingMask = (shield * 4) - std::abs(rank - backRank) * 2;
    const int endgameMask = (6 - centerDistance);
    const bool endgame = (whiteMajor + blackMajor) <= 2;
    return endgame ? endgameMask : openingMask;
  };

  score += kingSafetyMask(whiteKingSq, true) * params.kingSafetyPhaseMaskBonus;
  score -= kingSafetyMask(blackKingSq, false) * params.kingSafetyPhaseMaskBonus;

  const bool endgame = (whiteMajor + blackMajor) <= 2;
  if (endgame) {
    auto kingActivity = [](int sq) { return 6 - (std::abs((sq % 8) - 3) + std::abs((sq / 8) - 3)); };
    if (whiteKingSq >= 0) score += kingActivity(whiteKingSq) * params.endgameKingActivityBonus;
    if (blackKingSq >= 0) score -= kingActivity(blackKingSq) * params.endgameKingActivityBonus;
  } else {
    score += (whiteMinor + whiteMajor) * params.openingMobilityBonus;
    score -= (blackMinor + blackMajor) * params.openingMobilityBonus;
  }

  score += b.whiteToMove ? params.tempoBonus : -params.tempoBonus;

  return b.whiteToMove ? score : -score;
}

std::string breakdown(const board::Board& b, const Params& params) {
  std::ostringstream out;
  out << "eval=" << evaluate(b, params) << " stm=" << (b.whiteToMove ? 'w' : 'b')
      << " bp=" << params.bishopPairBonus << " tempo=" << params.tempoBonus;
  return out.str();
}

}  // namespace eval
