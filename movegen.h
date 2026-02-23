#ifndef MOVEGEN_H
#define MOVEGEN_H

#include <string>
#include <vector>

#include "board.h"

namespace movegen {

struct Move {
  int from = -1;
  int to = -1;
  char promotion = '\0';

  bool operator==(const Move& other) const {
    return from == other.from && to == other.to && promotion == other.promotion;
  }

  bool operator!=(const Move& other) const { return !(*this == other); }

  std::string toUCI() const;
};

bool parseUCIMove(const std::string& text, Move& out);
std::vector<Move> generatePseudoLegal(const board::Board& b);
std::vector<Move> generateLegal(const board::Board& b);
bool isLegalMove(const board::Board& b, const Move& m);

}  // namespace movegen

#endif
