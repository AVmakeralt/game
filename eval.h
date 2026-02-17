#ifndef EVAL_H
#define EVAL_H

#include <string>

#include "board.h"

namespace eval {

struct Params {
  int piece[6] = {100, 320, 330, 500, 900, 0};
};

void initialize(Params& p);
int evaluate(const board::Board& b, const Params& p);
std::string breakdown(const board::Board& b, const Params& p);

}  // namespace eval

#endif
