#ifndef BOARD_H
#define BOARD_H

#include <array>
#include <cctype>
#include <sstream>
#include <string>
#include <vector>

namespace board {

struct Board {
  std::array<char, 64> squares{};
  bool whiteToMove = true;
  std::vector<std::string> history;

  void clear() {
    squares.fill('.');
    whiteToMove = true;
    history.clear();
  }

  void setStartPos() {
    setFromFEN("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
  }

  bool setFromFEN(const std::string& fen) {
    clear();
    std::istringstream iss(fen);
    std::string placement;
    std::string side;
    if (!(iss >> placement >> side)) {
      return false;
    }

    int rank = 7;
    int file = 0;
    for (char c : placement) {
      if (c == '/') {
        --rank;
        file = 0;
        continue;
      }
      if (std::isdigit(static_cast<unsigned char>(c))) {
        file += (c - '0');
        continue;
      }
      if (rank < 0 || file > 7) {
        return false;
      }
      squares[rank * 8 + file] = c;
      ++file;
    }

    whiteToMove = (side == "w");
    history.clear();
    return true;
  }

  static int squareIndex(char fileChar, char rankChar) {
    if (fileChar < 'a' || fileChar > 'h' || rankChar < '1' || rankChar > '8') {
      return -1;
    }
    int file = fileChar - 'a';
    int rank = rankChar - '1';
    return rank * 8 + file;
  }

  char pieceAt(int idx) const {
    if (idx < 0 || idx >= 64) {
      return '.';
    }
    return squares[idx];
  }

  bool applyMove(int from, int to, char promotion = '\0') {
    if (from < 0 || from >= 64 || to < 0 || to >= 64) {
      return false;
    }
    const char piece = squares[from];
    if (piece == '.') {
      return false;
    }
    if (whiteToMove != static_cast<bool>(std::isupper(static_cast<unsigned char>(piece)))) {
      return false;
    }

    squares[to] = (promotion == '\0') ? piece : (whiteToMove ? static_cast<char>(std::toupper(promotion))
                                                               : static_cast<char>(std::tolower(promotion)));
    squares[from] = '.';
    whiteToMove = !whiteToMove;
    return true;
  }
};

}  // namespace board

#endif
