#ifndef EVAL_H
#define EVAL_H

namespace eval {
struct Params {
  int pawn = 100;
  int knight = 320;
  int bishop = 330;
  int rook = 500;
  int queen = 900;
};

inline void initialize(Params&) {}
}  // namespace eval

#endif
