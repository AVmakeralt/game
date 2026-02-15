#include "eval.h"

#include <array>
#include <cctype>
#include <sstream>

namespace eval {

void initialize(Params& p) { (void)p; }

static int pst(char piece, int sq) {
  static const std::array<int,64> knight = {
    -50,-40,-30,-30,-30,-30,-40,-50,-40,-20,0,5,5,0,-20,-40,-30,5,10,15,15,10,5,-30,
    -30,0,15,20,20,15,0,-30,-30,5,15,20,20,15,5,-30,-30,0,10,15,15,10,0,-30,
    -40,-20,0,0,0,0,-20,-40,-50,-40,-30,-30,-30,-30,-40,-50};
  char p = static_cast<char>(std::tolower(static_cast<unsigned char>(piece)));
  if (p == 'n') return knight[sq];
  return 0;
}

int evaluate(const board::Board& b, const Params& p) {
  int score = 0;
  int wBishops = 0, blBishops = 0;
  for (int sq = 0; sq < 64; ++sq) {
    char c = b.squares[sq];
    if (c == '.') continue;
    bool w = std::isupper(static_cast<unsigned char>(c));
    char q = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    int idx = q=='p'?0:q=='n'?1:q=='b'?2:q=='r'?3:q=='q'?4:5;
    int val = p.piece[idx];
    int psq = pst(c, w ? sq : (56 ^ sq));
    if (w) score += val + psq;
    else score -= val + psq;
    if (c == 'B') ++wBishops;
    if (c == 'b') ++blBishops;
  }
  if (wBishops >= 2) score += 30;
  if (blBishops >= 2) score -= 30;
  return b.whiteToMove ? score : -score;
}

std::string breakdown(const board::Board& b, const Params& p) {
  std::ostringstream oss;
  oss << "eval=" << evaluate(b, p) << " stm=" << (b.whiteToMove ? 'w' : 'b');
  return oss.str();
}

}  // namespace eval
