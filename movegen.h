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

inline void addPawnMoveWithPromotions(std::vector<Move>& out, int from, int to, bool isWhitePiece) {
  const int toRank = to / 8;
  if ((isWhitePiece && toRank == 7) || (!isWhitePiece && toRank == 0)) {
    out.push_back(Move{from, to, 'q'});
    out.push_back(Move{from, to, 'r'});
    out.push_back(Move{from, to, 'b'});
    out.push_back(Move{from, to, 'n'});
    return;
  }
  out.push_back(Move{from, to, '\0'});
}

inline int findKing(const board::Board& b, bool whiteKing) {
  const char king = whiteKing ? 'K' : 'k';
  for (int sq = 0; sq < 64; ++sq) {
    if (b.squares[static_cast<std::size_t>(sq)] == king) return sq;
  }
  return -1;
}

inline bool isSquareAttacked(const board::Board& b, int sq, bool byWhite) {
  if (sq < 0 || sq >= 64) return false;
  const int f = sq % 8;
  const int r = sq / 8;

  const int pawnDr = byWhite ? -1 : 1;
  for (int df : {-1, 1}) {
    int nf = f + df;
    int nr = r + pawnDr;
    if (nf < 0 || nf > 7 || nr < 0 || nr > 7) continue;
    char p = b.squares[static_cast<std::size_t>(nr * 8 + nf)];
    if (p == (byWhite ? 'P' : 'p')) return true;
  }

  constexpr int knightD[8][2] = {{1,2},{2,1},{2,-1},{1,-2},{-1,-2},{-2,-1},{-2,1},{-1,2}};
  for (const auto& d : knightD) {
    int nf = f + d[0], nr = r + d[1];
    if (nf < 0 || nf > 7 || nr < 0 || nr > 7) continue;
    char p = b.squares[static_cast<std::size_t>(nr * 8 + nf)];
    if (p == (byWhite ? 'N' : 'n')) return true;
  }

  auto ray = [&](int df, int dr, const char* sliders) {
    int nf = f + df, nr = r + dr;
    while (nf >= 0 && nf <= 7 && nr >= 0 && nr <= 7) {
      char p = b.squares[static_cast<std::size_t>(nr * 8 + nf)];
      if (p != '.') {
        if (static_cast<bool>(std::isupper(static_cast<unsigned char>(p))) == byWhite) {
          const char lower = static_cast<char>(std::tolower(static_cast<unsigned char>(p)));
          for (int i = 0; sliders[i]; ++i) if (lower == sliders[i]) return true;
        }
        break;
      }
      nf += df;
      nr += dr;
    }
    return false;
  };

  if (ray(1,0,"rq") || ray(-1,0,"rq") || ray(0,1,"rq") || ray(0,-1,"rq")) return true;
  if (ray(1,1,"bq") || ray(1,-1,"bq") || ray(-1,1,"bq") || ray(-1,-1,"bq")) return true;

  for (int df = -1; df <= 1; ++df) for (int dr = -1; dr <= 1; ++dr) if (df || dr) {
    int nf = f + df, nr = r + dr;
    if (nf < 0 || nf > 7 || nr < 0 || nr > 7) continue;
    char p = b.squares[static_cast<std::size_t>(nr * 8 + nf)];
    if (p == (byWhite ? 'K' : 'k')) return true;
  }

  return false;
}

inline bool inCheck(const board::Board& b, bool whiteKing) {
  int kingSq = findKing(b, whiteKing);
  if (kingSq < 0) return false;
  return isSquareAttacked(b, kingSq, !whiteKing);
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
        addPawnMoveWithPromotions(moves, from, one, isWhitePiece);
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
          addPawnMoveWithPromotions(moves, from, to, isWhitePiece);
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

inline std::vector<Move> generateLegal(const board::Board& b) {
  std::vector<Move> legal;
  auto candidates = generatePseudoLegal(b);
  legal.reserve(candidates.size());
  for (const auto& mv : candidates) {
    board::Board copy = b;
    if (!copy.applyMove(mv.from, mv.to, mv.promotion)) continue;
    if (!inCheck(copy, !copy.whiteToMove)) {
      legal.push_back(mv);
    }
  }
  return legal;
}

inline bool isLegalMove(const board::Board& b, const Move& m) {
  if (m.from < 0 || m.to < 0) return false;
  board::Board copy = b;
  if (!copy.applyMove(m.from, m.to, m.promotion)) return false;
  if (inCheck(copy, !copy.whiteToMove)) return false;

  const char piece = b.squares[static_cast<std::size_t>(m.from)];
  if (std::tolower(static_cast<unsigned char>(piece)) == 'p') {
    const int toRank = m.to / 8;
    const bool mustPromote = (std::isupper(static_cast<unsigned char>(piece)) && toRank == 7) ||
                             (std::islower(static_cast<unsigned char>(piece)) && toRank == 0);
    if (mustPromote && m.promotion == '\0') return false;
  }
  return true;
}

}  // namespace movegen

#endif
