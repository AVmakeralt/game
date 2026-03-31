#include "movegen.h"

#include <algorithm>
#include <cctype>

namespace movegen {

std::string Move::toUCI() const {
  if (from < 0 || to < 0) return "0000";
  std::string out;
  out.push_back(static_cast<char>('a' + from % 8));
  out.push_back(static_cast<char>('1' + from / 8));
  out.push_back(static_cast<char>('a' + to % 8));
  out.push_back(static_cast<char>('1' + to / 8));
  if (promotion) out.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(promotion))));
  return out;
}

bool parseUCIMove(const std::string& text, Move& out) {
  if (text.size() < 4) return false;
  out.from = board::Board::squareIndex(text[0], text[1]);
  out.to = board::Board::squareIndex(text[2], text[3]);
  out.promotion = text.size() >= 5 ? text[4] : '\0';
  return out.from >= 0 && out.to >= 0;
}

static bool sameSide(char a, char b) {
  if (a == '.' || b == '.') return false;
  return std::isupper(static_cast<unsigned char>(a)) == std::isupper(static_cast<unsigned char>(b));
}

static bool tryAddLegalMove(board::Board& copy, std::vector<Move>& legal, const Move& m) {
  board::Undo u;
  if (!copy.makeMove(m.from, m.to, m.promotion, u)) return false;
  legal.push_back(m);
  copy.unmakeMove(m.from, m.to, m.promotion, u);
  return true;
}

std::vector<Move> generateLegal(const board::Board& b) {
  std::vector<Move> legal;
  legal.reserve(96);
  static constexpr int pawnCaptureDf[2] = {-1, 1};
  board::Board copy = b;
  for (int from = 0; from < 64; ++from) {
    const char piece = b.squares[from];
    if (piece == '.') continue;
    const bool white = std::isupper(static_cast<unsigned char>(piece));
    if (white != b.whiteToMove) continue;
    const int r = from / 8;
    const int f = from % 8;
    const char p = static_cast<char>(std::tolower(static_cast<unsigned char>(piece)));

    if (p == 'p') {
      const int dir = white ? 1 : -1;
      const int one = from + dir * 8;
      if (one >= 0 && one < 64 && b.squares[one] == '.') {
        const bool promote = (white && one / 8 == 7) || (!white && one / 8 == 0);
        if (!promote) {
          tryAddLegalMove(copy, legal, {from, one, '\0'});
        } else {
          tryAddLegalMove(copy, legal, {from, one, 'q'});
          tryAddLegalMove(copy, legal, {from, one, 'r'});
          tryAddLegalMove(copy, legal, {from, one, 'b'});
          tryAddLegalMove(copy, legal, {from, one, 'n'});
        }
        const int two = from + dir * 16;
        if ((white ? r == 1 : r == 6) && b.squares[two] == '.') tryAddLegalMove(copy, legal, {from, two, '\0'});
      }
      for (int df : pawnCaptureDf) {
        const int nf = f + df;
        if (nf < 0 || nf > 7) continue;
        const int to = one + df;
        if (to < 0 || to >= 64) continue;
        if ((b.squares[to] != '.' && !sameSide(piece, b.squares[to])) || to == b.enPassantSquare) {
          const bool promote = (white && to / 8 == 7) || (!white && to / 8 == 0);
          if (!promote) {
            tryAddLegalMove(copy, legal, {from, to, '\0'});
          } else {
            tryAddLegalMove(copy, legal, {from, to, 'q'});
            tryAddLegalMove(copy, legal, {from, to, 'r'});
            tryAddLegalMove(copy, legal, {from, to, 'b'});
            tryAddLegalMove(copy, legal, {from, to, 'n'});
          }
        }
      }
      continue;
    }

    if (p == 'n') {
      static const int d[8][2] = {{1,2},{2,1},{2,-1},{1,-2},{-1,-2},{-2,-1},{-2,1},{-1,2}};
      for (auto& k : d) {
        const int nf = f + k[0], nr = r + k[1];
        if (nf < 0 || nf > 7 || nr < 0 || nr > 7) continue;
        const int to = nr * 8 + nf;
        if (!sameSide(piece, b.squares[to])) tryAddLegalMove(copy, legal, {from, to, '\0'});
      }
      continue;
    }

    auto slide = [&](const int dirs[][2], int count, bool single) {
      for (int i = 0; i < count; ++i) {
        int nf = f + dirs[i][0], nr = r + dirs[i][1];
        while (nf >= 0 && nf < 8 && nr >= 0 && nr < 8) {
          const int to = nr * 8 + nf;
          if (sameSide(piece, b.squares[to])) break;
          tryAddLegalMove(copy, legal, {from, to, '\0'});
          if (b.squares[to] != '.' || single) break;
          nf += dirs[i][0]; nr += dirs[i][1];
        }
      }
    };

    static const int bdirs[4][2] = {{1,1},{1,-1},{-1,1},{-1,-1}};
    static const int rdirs[4][2] = {{1,0},{-1,0},{0,1},{0,-1}};
    static const int kdirs[8][2] = {{1,1},{1,0},{1,-1},{0,1},{0,-1},{-1,1},{-1,0},{-1,-1}};

    if (p == 'b') slide(bdirs, 4, false);
    else if (p == 'r') slide(rdirs, 4, false);
    else if (p == 'q') { slide(bdirs, 4, false); slide(rdirs, 4, false); }
    else if (p == 'k') {
      slide(kdirs, 8, true);
      if (white) {
        if ((b.castlingRights & 1) && b.squares[5] == '.' && b.squares[6] == '.' &&
            !b.isSquareAttacked(4, false) && !b.isSquareAttacked(5, false) && !b.isSquareAttacked(6, false)) {
          tryAddLegalMove(copy, legal, {4, 6, '\0'});
        }
        if ((b.castlingRights & 2) && b.squares[3] == '.' && b.squares[2] == '.' && b.squares[1] == '.' &&
            !b.isSquareAttacked(4, false) && !b.isSquareAttacked(3, false) && !b.isSquareAttacked(2, false)) {
          tryAddLegalMove(copy, legal, {4, 2, '\0'});
        }
      } else {
        if ((b.castlingRights & 4) && b.squares[61] == '.' && b.squares[62] == '.' &&
            !b.isSquareAttacked(60, true) && !b.isSquareAttacked(61, true) && !b.isSquareAttacked(62, true)) {
          tryAddLegalMove(copy, legal, {60, 62, '\0'});
        }
        if ((b.castlingRights & 8) && b.squares[59] == '.' && b.squares[58] == '.' && b.squares[57] == '.' &&
            !b.isSquareAttacked(60, true) && !b.isSquareAttacked(59, true) && !b.isSquareAttacked(58, true)) {
          tryAddLegalMove(copy, legal, {60, 58, '\0'});
        }
      }
    }
  }
  return legal;
}

bool isLegalMove(const board::Board& b, const Move& m) {
  const auto legal = generateLegal(b);
  return std::find(legal.begin(), legal.end(), m) != legal.end();
}

}  // namespace movegen
