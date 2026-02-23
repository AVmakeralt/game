#ifndef TT_H
#define TT_H

#include <cstddef>
#include <cstdint>
#include <vector>

#include "board.h"

namespace tt {

enum class Bound : std::uint8_t { Exact = 0, Lower = 1, Upper = 2 };

struct Entry {
  std::uint64_t key = 0;
  int depth = -1;
  int score = 0;
  Bound bound = Bound::Exact;
  std::uint8_t generation = 0;
};

struct Table {
  std::vector<Entry> entries;
  std::uint8_t generation = 0;

  void initialize(std::size_t mb) {
    std::size_t bytes = mb * 1024ULL * 1024ULL;
    std::size_t count = bytes / sizeof(Entry);
    if (count == 0) count = 1;
    entries.assign(count, Entry{});
  }

  void clear() { entries.assign(entries.size(), Entry{}); }

  void nextGeneration() { ++generation; }

  bool probe(std::uint64_t key, Entry& out) const {
    if (entries.empty()) return false;
    const Entry& e = entries[static_cast<std::size_t>(key % entries.size())];
    if (e.key != key || e.depth < 0) return false;
    out = e;
    return true;
  }

  void store(std::uint64_t key, int depth, int score, Bound bound) {
    if (entries.empty()) return;
    Entry& slot = entries[static_cast<std::size_t>(key % entries.size())];
    const bool replace = (slot.key != key) || (depth >= slot.depth) || (slot.generation != generation);
    if (!replace) return;
    slot.key = key;
    slot.depth = depth;
    slot.score = score;
    slot.bound = bound;
    slot.generation = generation;
  }
};

void initializeZobrist();
std::uint64_t hash(const board::Board& b);

}  // namespace tt

#endif
