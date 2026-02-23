#ifndef BOARD_H
#define BOARD_H

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace board {

struct Undo {
  int prevEnPassant = -1;
  std::uint8_t prevCastling = 0;
  int prevHalfmove = 0;
  int prevFullmove = 1;
  bool prevWhiteToMove = true;
  char moved = '.';
  char captured = '.';
  int capturedSquare = -1;
  bool wasEnPassant = false;
  bool wasCastle = false;
  bool wasPromotion = false;
};

struct Board {
  std::array<char, 64> squares{};
  bool whiteToMove = true;
  std::uint8_t castlingRights = 0;
  int enPassantSquare = -1;
  int halfmoveClock = 0;
  int fullmoveNumber = 1;
  std::vector<std::string> history;

  void clear();
  void setStartPos();
  bool setFromFEN(const std::string& fen);

  static int squareIndex(char fileChar, char rankChar);
  static std::string squareName(int sq);

  char pieceAt(int idx) const;
  bool isSquareAttacked(int sq, bool byWhite) const;
  bool inCheck(bool white) const;
  bool makeMove(int from, int to, char promotion, Undo& u);
  void unmakeMove(int from, int to, char promotion, const Undo& u);

  bool applyMove(int from, int to, char promotion = '\0') {
    Undo u;
    return makeMove(from, to, promotion, u);
  }
};

}  // namespace board

#endif
