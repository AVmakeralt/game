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

  std::string toUCI() const;
  bool operator==(const Move& o) const { return from == o.from && to == o.to && promotion == o.promotion; }
};

bool parseUCIMove(const std::string& text, Move& out);
std::vector<Move> generatePseudoLegal(const board::Board& b);
std::vector<Move> generateLegal(const board::Board& b);
bool isLegalMove(const board::Board& b, const Move& m);

}  // namespace movegen

#endif
