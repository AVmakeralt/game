#include "tt.h"

#include <array>
#include <random>

namespace tt {

namespace {
std::array<std::array<std::uint64_t, 64>, 12> zPieces{};
std::array<std::uint64_t, 16> zCastle{};
std::array<std::uint64_t, 8> zEp{};
std::uint64_t zSide = 0;
bool initialized = false;

int pieceIndex(char p) {
  switch (p) {
    case 'P': return 0; case 'N': return 1; case 'B': return 2; case 'R': return 3; case 'Q': return 4; case 'K': return 5;
    case 'p': return 6; case 'n': return 7; case 'b': return 8; case 'r': return 9; case 'q': return 10; case 'k': return 11;
    default: return -1;
  }
}
}  // namespace

void initializeZobrist() {
  if (initialized) return;
  std::mt19937_64 rng(0xC0D3A5ULL);
  for (auto& piece : zPieces) for (auto& sq : piece) sq = rng();
  for (auto& x : zCastle) x = rng();
  for (auto& x : zEp) x = rng();
  zSide = rng();
  initialized = true;
}

std::uint64_t hash(const board::Board& b) {
  initializeZobrist();
  std::uint64_t h = 0;
  for (int sq = 0; sq < 64; ++sq) {
    int idx = pieceIndex(b.squares[sq]);
    if (idx >= 0) h ^= zPieces[idx][sq];
  }
  h ^= zCastle[b.castlingRights & 15u];
  if (b.enPassantSquare >= 0) h ^= zEp[b.enPassantSquare % 8];
  if (!b.whiteToMove) h ^= zSide;
  return h;
}

void Table::initialize(std::size_t mb) {
  std::size_t bytes = mb * 1024ull * 1024ull;
  std::size_t n = bytes / sizeof(Entry);
  if (n < 1) n = 1;
  entries_.assign(n, Entry{});
}

void Table::clear() {
  for (auto& e : entries_) e = Entry{};
}

bool Table::probe(std::uint64_t key, Entry& out) const {
  if (entries_.empty()) return false;
  const Entry& e = entries_[key % entries_.size()];
  if (e.depth >= 0 && e.key == key) {
    out = e;
    return true;
  }
  return false;
}

void Table::store(const Entry& e) {
  if (entries_.empty()) return;
  Entry& cur = entries_[e.key % entries_.size()];
  if (e.depth >= cur.depth || cur.key != e.key) cur = e;
}

}  // namespace tt
