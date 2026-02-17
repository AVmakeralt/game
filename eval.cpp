#include "eval.h"

#include <array>
#include <cctype>
#include <sstream>

#include "movegen.h"

namespace eval {

void initialize(Params& p) { (void)p; }

static int pst(char piece, int sq) {
  static const std::array<int, 64> knight = {
      -50, -40, -30, -30, -30, -30, -40, -50, -40, -20, 0, 5, 5, 0, -20, -40,
      -30, 5, 10, 15, 15, 10, 5, -30, -30, 0, 15, 20, 20, 15, 0, -30,
      -30, 5, 15, 20, 20, 15, 5, -30, -30, 0, 10, 15, 15, 10, 0, -30,
      -40, -20, 0, 0, 0, 0, -20, -40, -50, -40, -30, -30, -30, -30, -40, -50};
  static const std::array<int, 64> pawn = {
      0, 0, 0, 0, 0, 0, 0, 0, 50, 50, 50, 50, 50, 50, 50, 50,
      10, 10, 20, 30, 30, 20, 10, 10, 5, 5, 10, 25, 25, 10, 5, 5,
      0, 0, 0, 20, 20, 0, 0, 0, 5, -5, -10, 0, 0, -10, -5, 5,
      5, 10, 10, -20, -20, 10, 10, 5, 0, 0, 0, 0, 0, 0, 0, 0};

  char p = static_cast<char>(std::tolower(static_cast<unsigned char>(piece)));
  if (p == 'n') return knight[sq];
  if (p == 'p') return pawn[sq];
  return 0;
}

int evaluate(const board::Board& b, const Params& p) {
  int score = 0;
  int wBishops = 0, bBishops = 0;
  int wPawnsOnFile[8] = {0}, bPawnsOnFile[8] = {0};

  for (int sq = 0; sq < 64; ++sq) {
    char c = b.squares[sq];
    if (c == '.') continue;
    bool w = std::isupper(static_cast<unsigned char>(c));
    char q = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    int idx = q == 'p' ? 0 : q == 'n' ? 1 : q == 'b' ? 2 : q == 'r' ? 3 : q == 'q' ? 4 : 5;
    int val = p.piece[idx];
    int psq = pst(c, w ? sq : (56 ^ sq));
    if (w) score += val + psq;
    else score -= val + psq;
    if (c == 'B') ++wBishops;
    if (c == 'b') ++bBishops;
    if (c == 'P') ++wPawnsOnFile[sq % 8];
    if (c == 'p') ++bPawnsOnFile[sq % 8];
  }

  if (wBishops >= 2) score += 30;
  if (bBishops >= 2) score -= 30;

  for (int f = 0; f < 8; ++f) {
    if (wPawnsOnFile[f] > 1) score -= 12 * (wPawnsOnFile[f] - 1);
    if (bPawnsOnFile[f] > 1) score += 12 * (bPawnsOnFile[f] - 1);
  }

  board::Board copy = b;
  bool side = copy.whiteToMove;
  copy.whiteToMove = true;
  int wMobility = static_cast<int>(movegen::generateLegal(copy).size());
  copy.whiteToMove = false;
  int bMobility = static_cast<int>(movegen::generateLegal(copy).size());
  copy.whiteToMove = side;
  score += 3 * (wMobility - bMobility);

  int wKing = -1, bKing = -1;
  for (int i = 0; i < 64; ++i) {
    if (b.squares[i] == 'K') wKing = i;
    else if (b.squares[i] == 'k') bKing = i;
  }
  if (wKing >= 0) {
    int wr = wKing / 8;
    if (wr >= 1 && b.squares[wKing - 8] != 'P') score -= 15;
  }
  if (bKing >= 0) {
    int br = bKing / 8;
    if (br <= 6 && b.squares[bKing + 8] != 'p') score += 15;
  }

  return b.whiteToMove ? score : -score;
}

std::string breakdown(const board::Board& b, const Params& p) {
  std::ostringstream oss;
  oss << "eval=" << evaluate(b, p) << " stm=" << (b.whiteToMove ? 'w' : 'b');
  return oss.str();
}

}  // namespace eval
