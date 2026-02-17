#ifndef TT_H
#define TT_H

#include <cstddef>
#include <cstdint>
#include <vector>

#include "board.h"
#include "movegen.h"

namespace tt {

enum Bound : int { EXACT = 0, LOWER = 1, UPPER = 2 };

struct Entry {
  std::uint64_t key = 0;
  int depth = -1;
  int score = 0;
  Bound bound = EXACT;
  movegen::Move bestMove{};
  std::uint8_t age = 0;
};

class Table {
 public:
  void initialize(std::size_t mb);
  void clear();
  bool probe(std::uint64_t key, Entry& out);
  void store(const Entry& e);
  void newSearch();
  std::uint64_t probes() const { return probes_; }
  std::uint64_t hits() const { return hits_; }
  double hitRate() const { return probes_ ? static_cast<double>(hits_) / static_cast<double>(probes_) : 0.0; }

 private:
  std::vector<Entry> entries_;
  std::uint8_t generation_ = 0;
  std::uint64_t probes_ = 0;
  std::uint64_t hits_ = 0;
};

void initializeZobrist();
std::uint64_t hash(const board::Board& b);

}  // namespace tt

#endif
