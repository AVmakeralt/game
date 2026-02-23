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
    case 'P': return 0;
    case 'N': return 1;
    case 'B': return 2;
    case 'R': return 3;
    case 'Q': return 4;
    case 'K': return 5;
    case 'p': return 6;
    case 'n': return 7;
    case 'b': return 8;
    case 'r': return 9;
    case 'q': return 10;
    case 'k': return 11;
    default: return -1;
  }
}
}  // namespace

void initializeZobrist() {
  if (initialized) return;
  std::mt19937_64 rng(0xC0D3A5ULL);
  for (auto& piece : zPieces) {
    for (auto& sq : piece) sq = rng();
  }
  for (auto& x : zCastle) x = rng();
  for (auto& x : zEp) x = rng();
  zSide = rng();
  initialized = true;
}

std::uint64_t hash(const board::Board& b) {
  initializeZobrist();
  std::uint64_t h = 0;
  for (int sq = 0; sq < 64; ++sq) {
    const int idx = pieceIndex(b.squares[static_cast<std::size_t>(sq)]);
    if (idx >= 0) h ^= zPieces[static_cast<std::size_t>(idx)][static_cast<std::size_t>(sq)];
  }
  h ^= zCastle[static_cast<std::size_t>(b.castlingRights & 15u)];
  if (b.enPassantSquare >= 0) h ^= zEp[static_cast<std::size_t>(b.enPassantSquare % 8)];
  if (!b.whiteToMove) h ^= zSide;
  return h;
}

}  // namespace tt
