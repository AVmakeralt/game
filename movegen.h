#ifndef MOVEGEN_H
#define MOVEGEN_H

#include <cctype>
#include <string>
#include <vector>

#include "board.h"

namespace movegen {

struct Move {
  int from = -1;
  int to = -1;
  char promotion = '\0';

  std::string toUCI() const {
    if (from < 0 || to < 0) {
      return "0000";
    }
    std::string out;
    out.push_back(static_cast<char>('a' + (from % 8)));
    out.push_back(static_cast<char>('1' + (from / 8)));
    out.push_back(static_cast<char>('a' + (to % 8)));
    out.push_back(static_cast<char>('1' + (to / 8)));
    if (promotion != '\0') {
      out.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(promotion))));
    }
    return out;
  }
};

inline bool parseUCIMove(const std::string& text, Move& out) {
  if (text.size() < 4) {
    return false;
  }
  out.from = board::Board::squareIndex(text[0], text[1]);
  out.to = board::Board::squareIndex(text[2], text[3]);
  out.promotion = (text.size() >= 5) ? text[4] : '\0';
  return out.from >= 0 && out.to >= 0;
}

inline bool sameSide(char a, char b) {
  if (a == '.' || b == '.') return false;
  return static_cast<bool>(std::isupper(static_cast<unsigned char>(a))) ==
         static_cast<bool>(std::isupper(static_cast<unsigned char>(b)));
}

inline void addIfValid(const board::Board& b, int from, int to, std::vector<Move>& out) {
  if (to < 0 || to >= 64) return;
  char src = b.squares[from];
  char dst = b.squares[to];
  if (!sameSide(src, dst)) {
    out.push_back(Move{from, to, '\0'});
  }
}

inline std::vector<Move> generatePseudoLegal(const board::Board& b) {
  std::vector<Move> moves;
  for (int from = 0; from < 64; ++from) {
    const char piece = b.squares[from];
    if (piece == '.') continue;
    bool isWhitePiece = static_cast<bool>(std::isupper(static_cast<unsigned char>(piece)));
    if (isWhitePiece != b.whiteToMove) continue;

    int rank = from / 8;
    int file = from % 8;
    char p = static_cast<char>(std::tolower(static_cast<unsigned char>(piece)));

    if (p == 'p') {
      int dir = isWhitePiece ? 1 : -1;
      int one = from + 8 * dir;
      if (one >= 0 && one < 64 && b.squares[one] == '.') {
        moves.push_back(Move{from, one, '\0'});
        bool atStart = isWhitePiece ? (rank == 1) : (rank == 6);
        int two = from + 16 * dir;
        if (atStart && two >= 0 && two < 64 && b.squares[two] == '.') {
          moves.push_back(Move{from, two, '\0'});
        }
      }
      for (int df : {-1, 1}) {
        int nf = file + df;
        if (nf < 0 || nf > 7) continue;
        int to = one + df;
        if (to < 0 || to >= 64) continue;
        if (b.squares[to] != '.' && !sameSide(piece, b.squares[to])) {
          moves.push_back(Move{from, to, '\0'});
        }
      }
      continue;
    }

    if (p == 'n') {
      constexpr int kD[8][2] = {{1,2},{2,1},{2,-1},{1,-2},{-1,-2},{-2,-1},{-2,1},{-1,2}};
      for (auto& d : kD) {
        int nf = file + d[0], nr = rank + d[1];
        if (nf < 0 || nf > 7 || nr < 0 || nr > 7) continue;
        addIfValid(b, from, nr * 8 + nf, moves);
      }
      continue;
    }

    auto slide = [&](const std::vector<std::pair<int,int>>& dirs, bool singleStep) {
      for (const auto& [df, dr] : dirs) {
        int nf = file + df;
        int nr = rank + dr;
        while (nf >= 0 && nf <= 7 && nr >= 0 && nr <= 7) {
          int to = nr * 8 + nf;
          if (sameSide(piece, b.squares[to])) break;
          moves.push_back(Move{from, to, '\0'});
          if (b.squares[to] != '.') break;
          if (singleStep) break;
          nf += df;
          nr += dr;
        }
      }
    };

    if (p == 'b') slide({{1,1},{1,-1},{-1,1},{-1,-1}}, false);
    else if (p == 'r') slide({{1,0},{-1,0},{0,1},{0,-1}}, false);
    else if (p == 'q') slide({{1,1},{1,-1},{-1,1},{-1,-1},{1,0},{-1,0},{0,1},{0,-1}}, false);
    else if (p == 'k') slide({{1,1},{1,-1},{-1,1},{-1,-1},{1,0},{-1,0},{0,1},{0,-1}}, true);
  }
  return moves;
}

inline bool isLegalMove(const board::Board& b, const Move& m) {
  if (m.from < 0 || m.to < 0) return false;
  auto legalish = generatePseudoLegal(b);
  for (const auto& mv : legalish) {
    if (mv.from == m.from && mv.to == m.to) return true;
  }
  return false;
}

}  // namespace movegen

#endif
