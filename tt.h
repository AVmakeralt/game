#ifndef TT_H
#define TT_H

#include <cstddef>
#include <vector>

namespace tt {

struct Entry {
  unsigned long long key = 0;
  int depth = 0;
  int score = 0;
};

struct Table {
  std::vector<Entry> entries;

  void initialize(std::size_t mb) {
    std::size_t bytes = mb * 1024ULL * 1024ULL;
    std::size_t count = bytes / sizeof(Entry);
    if (count == 0) {
      count = 1;
    }
    entries.assign(count, Entry{});
  }

  void clear() { entries.assign(entries.size(), Entry{}); }
};

}  // namespace tt

#endif
