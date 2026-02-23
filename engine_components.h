#ifndef ENGINE_COMPONENTS_H
#define ENGINE_COMPONENTS_H

#include <algorithm>
#include <array>
#include <cstdint>
#include <cmath>
#include <thread>
#include <queue>
#include <mutex>
#include <cstring>
#include <condition_variable>
#include <atomic>
#include <fstream>
#include <future>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "movegen.h"

namespace engine_components {

namespace representation {
using Bitboard64 = std::uint64_t;
struct Bitboard128 {
  std::uint64_t lo = 0;
  std::uint64_t hi = 0;
};

struct TemporalBitboard {
  std::array<std::uint64_t, 4> history{};

  void push(std::uint64_t occ) {
    for (std::size_t i = history.size() - 1; i > 0; --i) history[i] = history[i - 1];
    history[0] = occ;
  }

  std::uint64_t velocityMask() const {
    return history[0] ^ history[1] ^ history[2] ^ history[3];
  }
};

struct AttackTables {
  std::array<Bitboard64, 64> rookAttacks{};
  std::array<Bitboard64, 64> bishopAttacks{};
  void initialize() {}
};

struct MagicTables {
  bool enabled = true;
  void initialize() {}
};
}  // namespace representation

namespace hashing {
struct Zobrist {
  std::array<std::array<std::uint64_t, 64>, 12> pieceSquare{};
  std::uint64_t sideToMove = 0;
  void initialize() {}
};

struct RepetitionTracker {
  std::vector<std::uint64_t> keys;
  int fiftyMoveClock = 0;

  void push(std::uint64_t key) { keys.push_back(key); }
  void clear() {
    keys.clear();
    fiftyMoveClock = 0;
  }
  bool isThreefold(std::uint64_t key) const {
    int count = 0;
    for (std::uint64_t k : keys) {
      if (k == key) ++count;
    }
    return count >= 3;
  }
};
}  // namespace hashing

namespace search_helpers {
struct KillerTable {
  std::array<std::array<movegen::Move, 2>, 128> killer{};
};

struct HistoryHeuristic {
  std::array<std::array<int, 64>, 64> score{};
};

struct CounterMoveTable {
  std::array<std::array<movegen::Move, 64>, 64> counter{};
};

struct PVTable {
  std::array<std::array<movegen::Move, 128>, 128> pv{};
  std::array<int, 128> length{};
};

struct SEE {
  static int pieceValue(char p) {
    switch (static_cast<char>(std::tolower(static_cast<unsigned char>(p)))) {
      case 'p': return 100;
      case 'n': return 320;
      case 'b': return 330;
      case 'r': return 500;
      case 'q': return 900;
      case 'k': return 20000;
      default: return 0;
    }
  }

  int estimate(const movegen::Move& m, const std::array<char, 64>* squares = nullptr) const {
    if (!squares || m.from < 0 || m.to < 0 || m.from >= 64 || m.to >= 64) return 0;
    const char from = (*squares)[static_cast<std::size_t>(m.from)];
    const char to = (*squares)[static_cast<std::size_t>(m.to)];
    int gain = pieceValue(to) - pieceValue(from) / 8;
    if (m.promotion != '\0') gain += pieceValue(m.promotion) - 100;
    return gain;
  }
};

struct SearchResultCache {
  std::unordered_map<std::string, std::string> openingBestMoves;

  std::string get(const std::string& key) const {
    auto it = openingBestMoves.find(key);
    return it == openingBestMoves.end() ? "" : it->second;
  }

  void put(const std::string& key, const std::string& move) { openingBestMoves[key] = move; }

  bool save(const std::string& path) const {
    std::ofstream out(path);
    if (!out) return false;
    for (const auto& [k, v] : openingBestMoves) out << k << ' ' << v << '\n';
    return true;
  }

  bool load(const std::string& path) {
    openingBestMoves.clear();
    std::ifstream in(path);
    if (!in) return false;
    std::string k;
    std::string v;
    while (in >> k >> v) openingBestMoves[k] = v;
    return true;
  }
};
}  // namespace search_helpers

namespace eval_model {
struct Handcrafted {
  int material = 0;
  int psqt = 0;
  int pawnStructure = 0;
  int kingSafety = 0;
  int mobility = 0;
  int space = 0;
  int bishopPair = 0;
  int rookActivity = 0;
  int tropism = 0;
  int tempo = 0;
  int initiative = 0;
  int timeAwareness = 0;

  int score() const {
    return material + psqt + pawnStructure + kingSafety + mobility + space + bishopPair + rookActivity + tropism +
           tempo + initiative + timeAwareness;
  }

  std::string breakdown() const {
    std::ostringstream oss;
    oss << "material=" << material << " psqt=" << psqt << " pawn=" << pawnStructure << " king=" << kingSafety
        << " mobility=" << mobility << " space=" << space << " bishopPair=" << bishopPair
        << " rookActivity=" << rookActivity << " tropism=" << tropism << " tempo=" << tempo
        << " initiative=" << initiative << " timeAwareness=" << timeAwareness;
    return oss.str();
  }
};

struct EndgameHeuristics {
  bool enabled = true;
  int kingPawnPattern = 0;
  int specializedScore = 0;
  int evaluate(bool deepEndgame) const { return deepEndgame ? kingPawnPattern + specializedScore : 0; }
};

struct NNUEConfig {
  int inputs = 2048;      // extended HalfKP + threat features
  int hidden1 = 3072;     // larger accumulator target range
  int hidden2 = 1024;     // post-accumulator mixer
  bool useSCReLU = true;  // squared clipped ReLU in first hidden layer
  bool useAMXPath = false;
  int draftHidden1 = 512; // tiny fast path width for lazy evaluation
  int miniQSearchHidden = 256;
  float policyPruneFloor = 0.05f;
};

struct NNUE {
  bool enabled = true;
  std::string weightsPath = "nnue.bin";
  NNUEConfig cfg{};
  std::vector<float> w1;
  std::vector<float> b1;
  std::vector<float> w2;
  std::vector<float> b2;
  std::vector<float> w3;
  float b3 = 0.0f;

  struct Accumulator {
    std::vector<float> hidden1;
    std::vector<float> features;
    bool initialized = false;
  };

  std::size_t parameterCount() const {
    return static_cast<std::size_t>(cfg.inputs) * cfg.hidden1 + static_cast<std::size_t>(cfg.hidden1) +
           static_cast<std::size_t>(cfg.hidden1) * cfg.hidden2 + static_cast<std::size_t>(cfg.hidden2) +
           static_cast<std::size_t>(cfg.hidden2) + 1;
  }

  void initializeWeights() {
    w1.assign(static_cast<std::size_t>(cfg.inputs * cfg.hidden1), 0.0f);
    b1.assign(static_cast<std::size_t>(cfg.hidden1), 0.0f);
    w2.assign(static_cast<std::size_t>(cfg.hidden1 * cfg.hidden2), 0.0f);
    b2.assign(static_cast<std::size_t>(cfg.hidden2), 0.0f);
    w3.assign(static_cast<std::size_t>(cfg.hidden2), 0.0f);

    for (std::size_t i = 0; i < w1.size(); ++i) w1[i] = static_cast<float>((static_cast<int>(i % 31) - 15) * 0.002f);
    for (std::size_t i = 0; i < w2.size(); ++i) w2[i] = static_cast<float>((static_cast<int>(i % 19) - 9) * 0.003f);
    for (std::size_t i = 0; i < w3.size(); ++i) w3[i] = static_cast<float>((static_cast<int>(i % 11) - 5) * 0.01f);
  }

  bool load(const std::string& path) {
    weightsPath = path;
    initializeWeights();

    std::ifstream in(path, std::ios::binary);
    if (in) {
      in.read(reinterpret_cast<char*>(w1.data()), static_cast<std::streamsize>(w1.size() * sizeof(float)));
      in.read(reinterpret_cast<char*>(b1.data()), static_cast<std::streamsize>(b1.size() * sizeof(float)));
      in.read(reinterpret_cast<char*>(w2.data()), static_cast<std::streamsize>(w2.size() * sizeof(float)));
      in.read(reinterpret_cast<char*>(b2.data()), static_cast<std::streamsize>(b2.size() * sizeof(float)));
      in.read(reinterpret_cast<char*>(w3.data()), static_cast<std::streamsize>(w3.size() * sizeof(float)));
      in.read(reinterpret_cast<char*>(&b3), static_cast<std::streamsize>(sizeof(b3)));
    }

    enabled = true;
    return true;
  }

  static std::vector<float> extractFeatures(const std::array<char, 64>& squares, bool whiteToMove, int inputSize) {
    std::vector<float> f(static_cast<std::size_t>(inputSize), 0.0f);
    auto piecePlane = [](char p) {
      switch (p) {
        case 'P': return 0; case 'N': return 1; case 'B': return 2; case 'R': return 3; case 'Q': return 4; case 'K': return 5;
        case 'p': return 6; case 'n': return 7; case 'b': return 8; case 'r': return 9; case 'q': return 10; case 'k': return 11;
        default: return -1;
      }
    };
    for (int sq = 0; sq < 64; ++sq) {
      int plane = piecePlane(squares[static_cast<std::size_t>(sq)]);
      if (plane < 0) continue;
      int idx = plane * 64 + sq;
      if (idx >= 0 && idx < inputSize) f[static_cast<std::size_t>(idx)] = 1.0f;
    }
    if (inputSize > 768) f[768] = whiteToMove ? 1.0f : -1.0f;

    int whiteBishops = 0;
    int blackBishops = 0;
    int whiteRooks = 0;
    int blackRooks = 0;
    std::array<int, 8> whitePawns{};
    std::array<int, 8> blackPawns{};
    for (int sq = 0; sq < 64; ++sq) {
      const char c = squares[static_cast<std::size_t>(sq)];
      if (c == 'B') ++whiteBishops;
      if (c == 'b') ++blackBishops;
      if (c == 'R') ++whiteRooks;
      if (c == 'r') ++blackRooks;
      if (c == 'P') ++whitePawns[static_cast<std::size_t>(sq % 8)];
      if (c == 'p') ++blackPawns[static_cast<std::size_t>(sq % 8)];
    }

    if (inputSize > 769) f[769] = (whiteBishops >= 2 ? 1.0f : 0.0f) - (blackBishops >= 2 ? 1.0f : 0.0f);
    if (inputSize > 770) f[770] = (whiteRooks >= 2 ? 1.0f : 0.0f) - (blackRooks >= 2 ? 1.0f : 0.0f);

    auto pawnPenalty = [](const std::array<int, 8>& files) {
      float p = 0.0f;
      for (int file = 0; file < 8; ++file) {
        if (files[static_cast<std::size_t>(file)] > 1) p += 0.5f;
        const bool left = file > 0 && files[static_cast<std::size_t>(file - 1)] > 0;
        const bool right = file < 7 && files[static_cast<std::size_t>(file + 1)] > 0;
        if (files[static_cast<std::size_t>(file)] > 0 && !left && !right) p += 0.5f;
      }
      return p;
    };

    if (inputSize > 771) f[771] = pawnPenalty(blackPawns) - pawnPenalty(whitePawns);
    if (inputSize > 772) f[772] = whiteToMove ? 0.5f : -0.5f;

    auto addThreatSlice = [&](int offset, bool whiteSide) {
      int directAttacks = 0;
      int pinnedPieces = 0;
      int mobilitySquares = 0;
      int kingSq = -1;
      for (int sq = 0; sq < 64; ++sq) {
        const char c = squares[static_cast<std::size_t>(sq)];
        if (c == '.') continue;
        const bool isWhite = std::isupper(static_cast<unsigned char>(c));
        if (std::tolower(static_cast<unsigned char>(c)) == 'k' && isWhite == whiteSide) kingSq = sq;
      }
      for (int sq = 0; sq < 64; ++sq) {
        const char c = squares[static_cast<std::size_t>(sq)];
        if (c == '.') continue;
        const bool isWhite = std::isupper(static_cast<unsigned char>(c));
        if (isWhite != whiteSide) continue;
        const int rank = sq / 8;
        const int file = sq % 8;
        const char p = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));

        if (p == 'n') {
          constexpr int kD[8][2] = {{1,2},{2,1},{2,-1},{1,-2},{-1,-2},{-2,-1},{-2,1},{-1,2}};
          for (auto& d : kD) {
            const int nf = file + d[0];
            const int nr = rank + d[1];
            if (nf < 0 || nf > 7 || nr < 0 || nr > 7) continue;
            ++mobilitySquares;
            const int to = nr * 8 + nf;
            const char dst = squares[static_cast<std::size_t>(to)];
            if (dst != '.' && (std::isupper(static_cast<unsigned char>(dst)) != isWhite)) ++directAttacks;
          }
        }

        if (p == 'b' || p == 'r' || p == 'q') {
          const std::array<std::pair<int, int>, 8> dirs{{{1,0},{-1,0},{0,1},{0,-1},{1,1},{1,-1},{-1,1},{-1,-1}}};
          const int fromKingDist = kingSq >= 0 ? std::abs((kingSq % 8) - file) + std::abs((kingSq / 8) - rank) : 10;
          for (const auto& [df, dr] : dirs) {
            if ((p == 'b') && (df == 0 || dr == 0)) continue;
            if ((p == 'r') && (df != 0 && dr != 0)) continue;
            int nf = file + df;
            int nr = rank + dr;
            bool seenOwn = false;
            while (nf >= 0 && nf <= 7 && nr >= 0 && nr <= 7) {
              ++mobilitySquares;
              const int to = nr * 8 + nf;
              const char dst = squares[static_cast<std::size_t>(to)];
              if (dst == '.') {
                nf += df;
                nr += dr;
                continue;
              }
              if (std::isupper(static_cast<unsigned char>(dst)) != isWhite) ++directAttacks;
              if (std::isupper(static_cast<unsigned char>(dst)) == isWhite) {
                if (seenOwn && fromKingDist <= 2) ++pinnedPieces;
                seenOwn = true;
              }
              break;
            }
          }
        }
      }

      for (int i = 0; i < 1024; ++i) {
        const int idx = offset + i;
        if (idx >= inputSize) break;
        float v = 0.0f;
        if (i < 340) v = static_cast<float>(directAttacks) / 32.0f;
        else if (i < 680) v = static_cast<float>(pinnedPieces) / 8.0f;
        else v = static_cast<float>(mobilitySquares) / 128.0f;
        f[static_cast<std::size_t>(idx)] = whiteSide ? v : -v;
      }
    };

    if (inputSize > 1024) {
      addThreatSlice(1024, true);
    }
    if (inputSize > 2048) {
      addThreatSlice(1024 + 1024, false);
    }

    return f;
  }

  void initializeAccumulator(Accumulator& acc, const std::vector<float>& input) const {
    acc.features = input;
    acc.hidden1.assign(static_cast<std::size_t>(cfg.hidden1), 0.0f);
    for (int h = 0; h < cfg.hidden1; ++h) {
      float v = b1[static_cast<std::size_t>(h)];
      for (int i = 0; i < cfg.inputs; ++i) {
        if (input[static_cast<std::size_t>(i)] == 0.0f) continue;
        v += input[static_cast<std::size_t>(i)] * w1[static_cast<std::size_t>(h * cfg.inputs + i)];
      }
      acc.hidden1[static_cast<std::size_t>(h)] = v;
    }
    acc.initialized = true;
  }

  void updateAccumulator(Accumulator& acc, const std::vector<int>& toggledFeatures, const std::vector<float>& newValues) const {
    if (!acc.initialized || toggledFeatures.size() != newValues.size()) return;
    for (std::size_t k = 0; k < toggledFeatures.size(); ++k) {
      const int idx = toggledFeatures[k];
      if (idx < 0 || idx >= cfg.inputs) continue;
      const float prev = acc.features[static_cast<std::size_t>(idx)];
      const float next = newValues[k];
      const float delta = next - prev;
      if (delta == 0.0f) continue;
      acc.features[static_cast<std::size_t>(idx)] = next;
      for (int h = 0; h < cfg.hidden1; ++h) {
        acc.hidden1[static_cast<std::size_t>(h)] += delta * w1[static_cast<std::size_t>(h * cfg.inputs + idx)];
      }
    }
  }

  int evaluateDraft(const std::vector<float>& input) const {
    if (!enabled || input.empty() || w1.empty()) return 0;
    const int draft = std::max(64, std::min(cfg.hidden1, cfg.draftHidden1));
    float out = b3;
    for (int h = 0; h < draft; ++h) {
      float v = b1[static_cast<std::size_t>(h)];
      for (int i = 0; i < cfg.inputs; ++i) {
        if (input[static_cast<std::size_t>(i)] == 0.0f) continue;
        v += input[static_cast<std::size_t>(i)] * w1[static_cast<std::size_t>(h * cfg.inputs + i)];
      }
      const float a = std::max(0.0f, v);
      out += a * w3[static_cast<std::size_t>(h % cfg.hidden2)];
    }
    return static_cast<int>(std::lround(out * 64.0f));
  }

  int evaluateFromAccumulator(const Accumulator& acc) const {
    if (!enabled || !acc.initialized || acc.hidden1.empty() || w2.empty()) return 0;
    std::vector<float> h2(static_cast<std::size_t>(cfg.hidden2), 0.0f);
    for (int h = 0; h < cfg.hidden2; ++h) {
      float v = b2[static_cast<std::size_t>(h)];
      for (int i = 0; i < cfg.hidden1; ++i) {
        const float relu = std::max(0.0f, acc.hidden1[static_cast<std::size_t>(i)]);
        v += relu * w2[static_cast<std::size_t>(h * cfg.hidden1 + i)];
      }
      h2[static_cast<std::size_t>(h)] = std::max(0.0f, v);
    }
    float out = b3;
    for (int i = 0; i < cfg.hidden2; ++i) out += h2[static_cast<std::size_t>(i)] * w3[static_cast<std::size_t>(i)];
    return static_cast<int>(std::lround(out * 100.0f));
  }

  int evaluateAMXKernel(const std::vector<float>& input) const {
#ifdef __APPLE__
    // Placeholder for Accelerate/AMX dispatch; keep equivalent semantics for now.
    Accumulator acc;
    initializeAccumulator(acc, input);
    return evaluateFromAccumulator(acc);
#else
    Accumulator acc;
    initializeAccumulator(acc, input);
    return evaluateFromAccumulator(acc);
#endif
  }

  int evaluate(const std::vector<float>& input) const {
    if (!enabled || input.empty() || w1.empty()) return 0;

    if (cfg.useAMXPath) {
      // Apple AMX/Accelerate backend hook.
      return evaluateAMXKernel(input);
    }

    std::vector<float> h1(static_cast<std::size_t>(cfg.hidden1), 0.0f);
    std::vector<float> h2(static_cast<std::size_t>(cfg.hidden2), 0.0f);

    for (int h = 0; h < cfg.hidden1; ++h) {
      float acc = b1[static_cast<std::size_t>(h)];
      for (int i = 0; i < cfg.inputs; ++i) acc += input[static_cast<std::size_t>(i)] * w1[static_cast<std::size_t>(h * cfg.inputs + i)];
      const float relu = std::max(0.0f, acc);
      if (cfg.useSCReLU) {
        const float clipped = std::min(1.0f, relu);
        h1[static_cast<std::size_t>(h)] = clipped * clipped;
      } else {
        h1[static_cast<std::size_t>(h)] = relu;
      }
    }
    for (int h = 0; h < cfg.hidden2; ++h) {
      float acc = b2[static_cast<std::size_t>(h)];
      for (int i = 0; i < cfg.hidden1; ++i) acc += h1[static_cast<std::size_t>(i)] * w2[static_cast<std::size_t>(h * cfg.hidden1 + i)];
      h2[static_cast<std::size_t>(h)] = std::max(0.0f, acc);
    }

    float out = b3;
    for (int i = 0; i < cfg.hidden2; ++i) out += h2[static_cast<std::size_t>(i)] * w3[static_cast<std::size_t>(i)];
    return static_cast<int>(std::lround(out * 100.0f));
  }

  int evaluateMiniQSearch(const std::vector<float>& input) const {
    if (!enabled || input.empty()) return 0;
    return evaluateDraft(input);
  }

  void distillStrategicHint(float policyActivation, float valueActivation) {
    if (w1.empty() || b1.empty()) return;
    const float blend = (policyActivation * 0.2f) + (valueActivation * 0.8f);
    const int taps = std::min(16, cfg.hidden1);
    for (int h = 0; h < taps; ++h) {
      b1[static_cast<std::size_t>(h)] += blend * 0.0005f;
    }
  }
};

struct StrategyConfig {
  int planes = 24;             // board feature planes
  int channels = 256;          // backbone channels
  int residualBlocks = 16;     // legacy residual mixer blocks
  int transformerHeads = 8;    // baseline self-attention heads
  int transformerLayers = 4;   // baseline token mixer layers
  int policyOutputs = 4096;    // from-square x to-square move lattice
  int topKForPruning = 5;      // search keeps only top-k policy moves
  float pruneThreshold = 0.90f;
  bool useHardPhaseSwitch = true;
  int activeExperts = 2;       // sparse top-k experts chosen by router
};

enum class GamePhase {
  Opening = 0,
  Middlegame = 1,
  Endgame = 2,
};

struct ExpertProfile {
  int transformerLayers = 2;
  int attentionHeads = 4;
  float policyBias = 0.0f;
};

struct StrategyRouterInput {
  float pieceCount = 0.0f;
  float materialBalance = 0.0f;
  float kingSafety = 0.0f;
};

struct StrategyOutput {
  int valueCp = 0;
  std::vector<float> policy;
  std::array<float, 3> wdl{{0.33f, 0.34f, 0.33f}};  // win/draw/loss
  std::array<float, 2> tacticalThreat{{0.0f, 0.0f}};
  std::array<float, 2> kingSafety{{0.0f, 0.0f}};
  std::array<float, 2> mobility{{0.0f, 0.0f}};
  std::array<float, 3> expertMix{{0.0f, 0.0f, 0.0f}};
};

struct StrategyNet {
  bool enabled = true;
  std::string weightsPath = "strategy_large.nn";
  StrategyConfig cfg{};
  std::array<ExpertProfile, 3> profiles{{
      ExpertProfile{2, 4, 0.15f},  // Theorist (opening): higher policy confidence
      ExpertProfile{4, 8, 0.05f},  // Tactician (middlegame): deepest stack
      ExpertProfile{3, 12, -0.05f} // Calculator (endgame): narrower but sharper focus
  }};
  std::vector<float> stem;
  std::vector<float> tokenProjection;
  std::vector<float> attentionQ;
  std::vector<float> attentionK;
  std::vector<float> attentionV;
  std::array<std::vector<float>, 3> expertBlocks;  // opening/mid/end experts
  std::array<std::vector<float>, 3> strategyBiasHead;  // long-term plan vectors
  std::vector<float> valueHead;
  std::array<float, 3> wdlHead{};
  std::array<float, 3> routerBias{{0.20f, 0.30f, 0.20f}};
  float valueBias = 0.0f;
  std::vector<float> policyHead;
  std::array<float, 2> tacticalHead{};
  std::array<float, 2> kingSafetyHead{};
  std::array<float, 2> mobilityHead{};

  std::size_t parameterCount() const {
    const std::size_t stemParams = static_cast<std::size_t>(cfg.planes) * cfg.channels;
    const std::size_t tokenParams = static_cast<std::size_t>(64) * cfg.channels;
    const std::size_t attentionParams = static_cast<std::size_t>(cfg.transformerLayers) * 3ULL * cfg.channels * cfg.channels;
    const std::size_t expertParams = 3ULL * static_cast<std::size_t>(cfg.residualBlocks) * cfg.channels * cfg.channels;
    const std::size_t strategyBiasParams = 3ULL * static_cast<std::size_t>(cfg.channels) * cfg.policyOutputs;
    const std::size_t headParams = static_cast<std::size_t>(cfg.channels) * (1 + cfg.policyOutputs) + cfg.policyOutputs + 3;
    return stemParams + tokenParams + attentionParams + expertParams + strategyBiasParams + headParams;
  }

  void initializeWeights() {
    stem.assign(static_cast<std::size_t>(cfg.planes * cfg.channels), 0.001f);
    tokenProjection.assign(static_cast<std::size_t>(64 * cfg.channels), 0.0008f);
    attentionQ.assign(static_cast<std::size_t>(cfg.transformerLayers * cfg.channels * cfg.channels), 0.0f);
    attentionK.assign(static_cast<std::size_t>(cfg.transformerLayers * cfg.channels * cfg.channels), 0.0f);
    attentionV.assign(static_cast<std::size_t>(cfg.transformerLayers * cfg.channels * cfg.channels), 0.0f);
    for (auto& expert : expertBlocks) {
      expert.assign(static_cast<std::size_t>(cfg.residualBlocks * cfg.channels * cfg.channels), 0.0f);
    }
    for (auto& strategyBias : strategyBiasHead) {
      strategyBias.assign(static_cast<std::size_t>(cfg.channels * cfg.policyOutputs), 0.0f);
    }
    valueHead.assign(static_cast<std::size_t>(cfg.channels), 0.0f);
    policyHead.assign(static_cast<std::size_t>(cfg.channels * cfg.policyOutputs), 0.0f);

    for (std::size_t i = 0; i < attentionQ.size(); ++i) attentionQ[i] = static_cast<float>((static_cast<int>(i % 29) - 14) * 0.0003f);
    for (std::size_t i = 0; i < attentionK.size(); ++i) attentionK[i] = static_cast<float>((static_cast<int>(i % 31) - 15) * 0.0003f);
    for (std::size_t i = 0; i < attentionV.size(); ++i) attentionV[i] = static_cast<float>((static_cast<int>(i % 19) - 9) * 0.0004f);
    for (int e = 0; e < 3; ++e) {
      for (std::size_t i = 0; i < expertBlocks[static_cast<std::size_t>(e)].size(); ++i) {
        expertBlocks[static_cast<std::size_t>(e)][i] = static_cast<float>((static_cast<int>((i + e) % 23) - 11) * 0.0005f);
      }
      for (std::size_t i = 0; i < strategyBiasHead[static_cast<std::size_t>(e)].size(); ++i) {
        strategyBiasHead[static_cast<std::size_t>(e)][i] = static_cast<float>((static_cast<int>((i + 2 * e) % 37) - 18) * 0.0008f);
      }
    }
    for (std::size_t i = 0; i < valueHead.size(); ++i) valueHead[i] = static_cast<float>((static_cast<int>(i % 13) - 6) * 0.01f);
    for (std::size_t i = 0; i < policyHead.size(); ++i) policyHead[i] = static_cast<float>((static_cast<int>(i % 17) - 8) * 0.0015f);
    wdlHead = {0.12f, 0.05f, -0.12f};
    tacticalHead = {0.15f, -0.15f};
    kingSafetyHead = {0.12f, -0.12f};
    mobilityHead = {0.08f, -0.08f};
  }

  bool load(const std::string& path) {
    weightsPath = path;
    initializeWeights();

    std::ifstream in(path, std::ios::binary);
    if (in) {
      in.read(reinterpret_cast<char*>(stem.data()), static_cast<std::streamsize>(stem.size() * sizeof(float)));
      in.read(reinterpret_cast<char*>(tokenProjection.data()), static_cast<std::streamsize>(tokenProjection.size() * sizeof(float)));
      in.read(reinterpret_cast<char*>(attentionQ.data()), static_cast<std::streamsize>(attentionQ.size() * sizeof(float)));
      in.read(reinterpret_cast<char*>(attentionK.data()), static_cast<std::streamsize>(attentionK.size() * sizeof(float)));
      in.read(reinterpret_cast<char*>(attentionV.data()), static_cast<std::streamsize>(attentionV.size() * sizeof(float)));
      for (auto& expert : expertBlocks) {
        in.read(reinterpret_cast<char*>(expert.data()), static_cast<std::streamsize>(expert.size() * sizeof(float)));
      }
      for (auto& strategyBias : strategyBiasHead) {
        in.read(reinterpret_cast<char*>(strategyBias.data()), static_cast<std::streamsize>(strategyBias.size() * sizeof(float)));
      }
      in.read(reinterpret_cast<char*>(valueHead.data()), static_cast<std::streamsize>(valueHead.size() * sizeof(float)));
      in.read(reinterpret_cast<char*>(&valueBias), static_cast<std::streamsize>(sizeof(valueBias)));
      in.read(reinterpret_cast<char*>(policyHead.data()), static_cast<std::streamsize>(policyHead.size() * sizeof(float)));
    }

    enabled = true;
    return true;
  }

  StrategyRouterInput computeRouterInput(const std::vector<float>& planes) const {
    StrategyRouterInput in{};
    if (planes.empty()) return in;
    for (float v : planes) {
      in.pieceCount += std::abs(v);
      in.materialBalance += v;
    }
    in.pieceCount = std::min(32.0f, in.pieceCount);
    in.materialBalance = std::clamp(in.materialBalance, -16.0f, 16.0f);
    in.kingSafety = planes.size() >= 2 ? (planes[0] - planes[1]) : 0.0f;
    return in;
  }

  std::array<float, 3> routeExperts(const StrategyRouterInput& in, GamePhase hint) const {
    std::array<float, 3> logits{
        0.07f * in.pieceCount - 0.02f * std::abs(in.materialBalance) + routerBias[0],
        0.04f * in.pieceCount + 0.05f * std::abs(in.kingSafety) + routerBias[1],
        0.09f * (32.0f - in.pieceCount) + 0.03f * std::abs(in.materialBalance) + routerBias[2]};

    if (cfg.useHardPhaseSwitch) {
      logits = {0.0f, 0.0f, 0.0f};
      logits[static_cast<std::size_t>(hint)] = 1.0f;
    }

    const float maxLogit = std::max({logits[0], logits[1], logits[2]});
    std::array<float, 3> probs{};
    float denom = 0.0f;
    for (int i = 0; i < 3; ++i) {
      probs[static_cast<std::size_t>(i)] = std::exp(logits[static_cast<std::size_t>(i)] - maxLogit);
      denom += probs[static_cast<std::size_t>(i)];
    }
    denom = std::max(1e-6f, denom);
    for (float& p : probs) p /= denom;

    std::array<int, 3> idx{0, 1, 2};
    std::sort(idx.begin(), idx.end(), [&](int a, int b) { return probs[static_cast<std::size_t>(a)] > probs[static_cast<std::size_t>(b)]; });
    std::array<float, 3> sparse{};
    const int keep = std::clamp(cfg.activeExperts, 1, 2);
    float sparseNorm = 0.0f;
    for (int i = 0; i < keep; ++i) {
      sparse[static_cast<std::size_t>(idx[static_cast<std::size_t>(i)])] = probs[static_cast<std::size_t>(idx[static_cast<std::size_t>(i)])];
      sparseNorm += sparse[static_cast<std::size_t>(idx[static_cast<std::size_t>(i)])];
    }
    sparseNorm = std::max(1e-6f, sparseNorm);
    for (float& p : sparse) p /= sparseNorm;
    return sparse;
  }

  StrategyOutput evaluate(const std::vector<float>& planes, GamePhase phase = GamePhase::Middlegame) const {
    StrategyOutput out;
    out.policy.assign(static_cast<std::size_t>(cfg.policyOutputs), 0.0f);
    if (!enabled || planes.empty() || stem.empty()) return out;

    std::vector<float> state(static_cast<std::size_t>(cfg.channels), 0.0f);
    for (int c = 0; c < cfg.channels; ++c) {
      float acc = 0.0f;
      for (int p = 0; p < cfg.planes; ++p) {
        const std::size_t wIdx = static_cast<std::size_t>(p * cfg.channels + c);
        acc += planes[static_cast<std::size_t>(p)] * stem[wIdx];
      }
      state[static_cast<std::size_t>(c)] = std::max(0.0f, acc);
    }

    for (int layer = 0; layer < cfg.transformerLayers; ++layer) {
      std::vector<float> q(static_cast<std::size_t>(cfg.channels), 0.0f);
      std::vector<float> k(static_cast<std::size_t>(cfg.channels), 0.0f);
      std::vector<float> v(static_cast<std::size_t>(cfg.channels), 0.0f);
      const std::size_t attnOffset = static_cast<std::size_t>(layer * cfg.channels * cfg.channels);
      for (int c = 0; c < cfg.channels; ++c) {
        for (int i = 0; i < cfg.channels; ++i) {
          const float s = state[static_cast<std::size_t>(i)];
          q[static_cast<std::size_t>(c)] += s * attentionQ[attnOffset + static_cast<std::size_t>(c * cfg.channels + i)];
          k[static_cast<std::size_t>(c)] += s * attentionK[attnOffset + static_cast<std::size_t>(c * cfg.channels + i)];
          v[static_cast<std::size_t>(c)] += s * attentionV[attnOffset + static_cast<std::size_t>(c * cfg.channels + i)];
        }
      }
      float qk = 0.0f;
      for (int c = 0; c < cfg.channels; ++c) qk += q[static_cast<std::size_t>(c)] * k[static_cast<std::size_t>(c)];
      const float attn = 1.0f / (1.0f + std::exp(-qk / std::max(1.0f, static_cast<float>(cfg.channels))));
      for (int c = 0; c < cfg.channels; ++c) {
        state[static_cast<std::size_t>(c)] = std::max(0.0f, state[static_cast<std::size_t>(c)] + v[static_cast<std::size_t>(c)] * attn);
      }
    }

    const auto routerIn = computeRouterInput(planes);
    const auto mix = routeExperts(routerIn, phase);
    out.expertMix = mix;

    std::vector<float> expertState(static_cast<std::size_t>(cfg.channels), 0.0f);
    for (int e = 0; e < 3; ++e) {
      const float weight = mix[static_cast<std::size_t>(e)];
      if (weight <= 0.0f) continue;
      std::vector<float> local = state;
      const auto& expert = expertBlocks[static_cast<std::size_t>(e)];
      const int depth = std::min(cfg.residualBlocks, profiles[static_cast<std::size_t>(e)].transformerLayers + cfg.residualBlocks / 2);
      for (int b = 0; b < depth; ++b) {
        std::vector<float> next = local;
        const std::size_t offset = static_cast<std::size_t>(b * cfg.channels * cfg.channels);
        for (int c = 0; c < cfg.channels; ++c) {
          float mixValue = 0.0f;
          for (int i = 0; i < cfg.channels; ++i) {
            mixValue += local[static_cast<std::size_t>(i)] * expert[offset + static_cast<std::size_t>(c * cfg.channels + i)];
          }
          next[static_cast<std::size_t>(c)] = std::max(0.0f, local[static_cast<std::size_t>(c)] + mixValue);
        }
        local.swap(next);
      }
      for (int c = 0; c < cfg.channels; ++c) {
        expertState[static_cast<std::size_t>(c)] += local[static_cast<std::size_t>(c)] * weight;
      }
    }

    float value = valueBias;
    for (int c = 0; c < cfg.channels; ++c) value += expertState[static_cast<std::size_t>(c)] * valueHead[static_cast<std::size_t>(c)];
    out.valueCp = static_cast<int>(std::lround(value * 100.0f));

    std::vector<float> strategyBias(static_cast<std::size_t>(cfg.policyOutputs), 0.0f);
    for (int e = 0; e < 3; ++e) {
      const float weight = mix[static_cast<std::size_t>(e)];
      if (weight <= 0.0f) continue;
      for (int m = 0; m < cfg.policyOutputs; ++m) {
        float biasLogit = 0.0f;
        for (int c = 0; c < cfg.channels; ++c) {
          const std::size_t wIdx = static_cast<std::size_t>(c * cfg.policyOutputs + m);
          biasLogit += expertState[static_cast<std::size_t>(c)] * strategyBiasHead[static_cast<std::size_t>(e)][wIdx];
        }
        strategyBias[static_cast<std::size_t>(m)] += biasLogit * weight;
      }
    }

    for (int m = 0; m < cfg.policyOutputs; ++m) {
      float logit = 0.0f;
      for (int c = 0; c < cfg.channels; ++c) {
        const std::size_t wIdx = static_cast<std::size_t>(c * cfg.policyOutputs + m);
        logit += expertState[static_cast<std::size_t>(c)] * policyHead[wIdx];
      }
      float profileBias = 0.0f;
      for (int e = 0; e < 3; ++e) {
        profileBias += mix[static_cast<std::size_t>(e)] * profiles[static_cast<std::size_t>(e)].policyBias;
      }
      out.policy[static_cast<std::size_t>(m)] = logit + strategyBias[static_cast<std::size_t>(m)] + profileBias;
    }

    const float winLogit = expertState[0] * wdlHead[0];
    const float drawLogit = expertState[std::min(1, cfg.channels - 1)] * wdlHead[1];
    const float lossLogit = expertState[std::min(2, cfg.channels - 1)] * wdlHead[2];
    const float maxWdl = std::max({winLogit, drawLogit, lossLogit});
    const float ew = std::exp(winLogit - maxWdl);
    const float ed = std::exp(drawLogit - maxWdl);
    const float el = std::exp(lossLogit - maxWdl);
    const float norm = std::max(1e-6f, ew + ed + el);
    out.wdl = {ew / norm, ed / norm, el / norm};

    const int whiteMaterialAnchor = 4;
    const int blackMaterialAnchor = std::max(8, cfg.channels / 3);
    const int mobilityAnchor = std::max(16, cfg.channels / 2);
    out.tacticalThreat[0] = expertState[static_cast<std::size_t>(whiteMaterialAnchor)] * tacticalHead[0];
    out.tacticalThreat[1] = expertState[static_cast<std::size_t>(blackMaterialAnchor)] * tacticalHead[1];
    out.kingSafety[0] = expertState[static_cast<std::size_t>(whiteMaterialAnchor + 1)] * kingSafetyHead[0];
    out.kingSafety[1] = expertState[static_cast<std::size_t>(blackMaterialAnchor + 1)] * kingSafetyHead[1];
    out.mobility[0] = expertState[static_cast<std::size_t>(mobilityAnchor)] * mobilityHead[0];
    out.mobility[1] = expertState[static_cast<std::size_t>(std::min(cfg.channels - 1, mobilityAnchor + 4))] * mobilityHead[1];

    return out;
  }
};

struct PolicyNet {
  bool enabled = false;
  std::vector<float> priors;
};

struct CATConfig {
  bool enabled = true;
  int lowBudgetNodes = 2000;
  int highBudgetNodes = 200000;
  float disagreementThreshold = 80.0f;
};

struct LossLearning {
  int lossCases = 0;
  int adversarialTests = 0;

  void recordLoss() { ++lossCases; }
  void runAdversarialSweep() { ++adversarialTests; }
};

struct TrainingInfra {
  bool selfPlayEnabled = false;
  bool supervisedEnabled = false;
  bool distillationEnabled = false;
  std::string replayBufferPath = "replay.bin";
  CATConfig cat;
  LossLearning lossLearning;
};
}  // namespace eval_model

namespace search_arch {
struct Features {
  bool usePVS = true;
  bool useAspiration = true;
  bool useQuiescence = true;
  bool useNullMove = true;
  bool useLMR = true;
  bool useFutility = true;
  bool useMateDistancePruning = true;
  bool useExtensions = true;
  bool useMultiPV = true;
  bool useMCTS = false;
  bool useParallel = false;
  bool useAsync = false;
  bool useMultiRateThinking = true;
  bool usePolicyPruning = true;
  bool usePolicyValuePruning = true;
  bool useLazyEval = true;
  int policyTopK = 5;
  float policyPruneThreshold = 0.90f;
  int masterEvalTopMoves = 3;
  int multiPV = 1;
};

struct ParallelConfig {
  int threads = 1;
  bool rootParallel = false;
  bool treeSplit = false;
  bool hashSync = false;
  bool loadBalancing = false;
  bool ybwcFirstMoveSerial = true;
  int splitDepthLimit = 8;
  int maxSplitMoves = 6;
  bool deterministicMode = false;
};

struct MCTSConfig {
  bool enabled = false;
  int simulations = 0;
  int miniBatchSize = 256;
  float virtualLoss = 0.25f;
  bool usePhaseAwareM2CTS = true;
  bool useCladeSelection = true;
  float fpuReduction = 0.20f;
};
}  // namespace search_arch

namespace opening {
struct Book {
  bool enabled = false;
  std::string format = "polyglot";
  std::string path = "book.bin";
  std::unordered_map<std::string, std::string> moveByKey;

  void seedDefaults() {
    if (!moveByKey.empty()) return;
    moveByKey["startpos"] = "e2e4";
    moveByKey["e2e4"] = "e7e5";
    moveByKey["d2d4"] = "d7d5";
  }

  std::string probe(const std::string& key) const {
    if (!enabled) return "";
    auto it = moveByKey.find(key);
    return it == moveByKey.end() ? "" : it->second;
  }
};

struct Novelty {
  std::unordered_map<std::string, int> seenPositions;
  bool isNovel(const std::string& key) {
    int& seen = seenPositions[key];
    ++seen;
    return seen == 1;
  }
};

struct BookBuilder {
  std::vector<std::pair<std::string, std::string>> lines;
  void addLine(const std::string& key, const std::string& move) { lines.push_back({key, move}); }
};

struct PrepModule {
  bool noveltySearch = false;
  bool pruneBook = false;
  Novelty novelty;
  BookBuilder builder;
};
}  // namespace opening

namespace timing {
struct Manager {
  int remainingMs = 0;
  int incrementMs = 0;
  std::string mode = "classical";
  bool pondering = false;
  int allocateMoveTimeMs(int fallback) const {
    if (remainingMs <= 0) return fallback;
    return remainingMs / 30 + incrementMs;
  }
};
}  // namespace timing

namespace tooling {
struct TrainingMetrics {
  float currentLoss = 0.0f;
  float eloGain = 0.0f;
  int nodesPerSecond = 0;
  std::array<char, 64> statusMsg{};
};

struct SharedMetricsIPC {
  std::string path = "training.ipc";
  TrainingMetrics last{};

  bool write(const TrainingMetrics& metrics) {
    last = metrics;
    return true;
  }
};

struct BinpackReader {
  std::string path = "selfplay.binpack";
  std::size_t estimatePositionThroughput() const { return 0; }
};

struct Formats {
  bool pgnEnabled = true;
  bool epdEnabled = true;
};

struct Integrity {
  bool antiCheatEnabled = false;
  bool checksumOk = true;
  bool verifyRuntime() const { return !antiCheatEnabled || checksumOk; }
};

struct RamTablebase {
  bool enabled = false;
  bool loaded = false;
  std::unordered_map<std::string, int> wdlByKey;

  void preload6ManMock() {
    loaded = true;
    wdlByKey["K1v0"] = 1;
    wdlByKey["K1v1"] = 1;
    wdlByKey["K2v1"] = 1;
  }

  int probe(const std::string& key) const {
    if (!enabled || !loaded) return 0;
    auto it = wdlByKey.find(key);
    return it == wdlByKey.end() ? 0 : it->second;
  }
};

struct TestHarness {
  bool regressionEnabled = true;
  bool eloEnabled = true;
  bool selfPlayTournaments = true;
  std::unordered_map<std::string, double> params;
  SharedMetricsIPC ipc;
  BinpackReader binpack;
};
}  // namespace tooling

}  // namespace engine_components

#endif
