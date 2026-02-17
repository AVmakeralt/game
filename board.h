#ifndef BOARD_H
#define BOARD_H

#include <array>
#include <string>
#include <vector>

namespace board {

struct Undo {
  std::array<char, 64> prevSquares{};
  bool prevWhiteToMove = true;
  unsigned prevCastling = 0;
  int prevEnPassant = -1;
  int prevHalfmove = 0;
  int prevFullmove = 1;
};

struct Board {
  std::array<char, 64> squares{};
  bool whiteToMove = true;
  unsigned castlingRights = 0;  // 1=K,2=Q,4=k,8=q
  int enPassantSquare = -1;
  int halfmoveClock = 0;
  int fullmoveNumber = 1;
  std::vector<std::string> history;

  void clear();
  void setStartPos();
  bool setFromFEN(const std::string& fen);
  static int squareIndex(char file, char rank);
  static std::string squareName(int sq);
  bool makeMove(int from, int to, char promotion, Undo& u);
  void unmakeMove(int from, int to, char promotion, const Undo& u);
  bool isSquareAttacked(int sq, bool byWhite) const;
  bool inCheck(bool white) const;
};

}  // namespace board

#endif
