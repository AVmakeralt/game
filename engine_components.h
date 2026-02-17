#ifndef ENGINE_COMPONENTS_H
#define ENGINE_COMPONENTS_H

#include <array>
#include <cstdint>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <algorithm>
#include <vector>

#include "movegen.h"

namespace engine_components {

namespace representation {
using Bitboard64 = std::uint64_t;
struct Bitboard128 {
  std::uint64_t lo = 0;
  std::uint64_t hi = 0;
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
  std::array<std::array<int, 64>, 64> success{};
  std::array<std::array<int, 64>, 64> fail{};

  int adaptiveScore(int from, int to) const {
    int s = score[from][to];
    int ok = success[from][to];
    int bad = fail[from][to];
    int ratio = ok * 100 / std::max(1, ok + bad);
    return s + ratio;
  }

  void recordSuccess(int from, int to, int bonus) {
    score[from][to] += bonus;
    success[from][to] += 1;
  }

  void recordFail(int from, int to) { fail[from][to] += 1; }
};

struct CounterMoveTable {
  std::array<std::array<movegen::Move, 64>, 64> counter{};
};

struct PVTable {
  std::array<std::array<movegen::Move, 128>, 128> pv{};
  std::array<int, 128> length{};
};

struct SEE {
  int estimate(const movegen::Move&) const { return 0; }
};

struct TacticalSolver {
  int forcingScore(const movegen::Move& m) const {
    int score = 0;
    if (m.promotion != '\0') score += 300;
    if (m.from >= 0 && m.to >= 0 && std::abs((m.to / 8) - (m.from / 8)) >= 2) score += 60;
    return score;
  }
};

struct DynamicPVWeights {
  std::unordered_map<std::string, int> weights;
  int get(const std::string& key) const {
    auto it = weights.find(key);
    return it == weights.end() ? 0 : it->second;
  }
  void reward(const std::string& key, int delta) { weights[key] += delta; }
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
  int opposition = 0;
  int triangulation = 0;
  int zugzwang = 0;

  int evaluate(bool deepEndgame) const {
    if (!deepEndgame) return 0;
    return kingPawnPattern + specializedScore + opposition + triangulation + zugzwang;
  }
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
  LossLearning lossLearning;
};
}  // namespace eval_model

namespace search_arch {
struct PhaseContext {
  int threatDensity = 0;
  int kingSafety = 0;
  int materialImbalance = 0;
  int pawnTropism = 0;
  bool endgame = false;
};

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
  bool useParallel = false;
  bool useAsync = false;
  bool useMultiRateThinking = true;
  bool useAdaptiveMoveOrdering = true;
  bool useAdaptiveLmr = true;
  bool usePhaseAwareFutility = true;
  bool useThreatExtensions = true;
  bool useTacticalPrefilter = true;
  bool useSelectiveDeepening = true;
  bool useIID = true;
  bool useLMP = true;
  bool useProbabilisticPruning = true;
  bool useMCTS = false;
  int multiPV = 1;
};

struct ParallelConfig {
  int threads = 1;
  bool rootParallel = false;
  bool treeSplit = false;
  bool hashSync = false;
  bool loadBalancing = false;
  bool numaAware = false;
};

struct MCTSConfig {
  bool enabled = false;
  int simulations = 0;
};
}  // namespace search_arch

namespace opening {
struct Book {
  bool enabled = false;
  std::string format = "polyglot";
  std::string path = "book.bin";

  std::string probe() const { return ""; }
};

struct DynamicWeights {
  std::unordered_map<std::string, int> lineScore;
  int weight(const std::string& key) const {
    auto it = lineScore.find(key);
    return it == lineScore.end() ? 0 : it->second;
  }
  void update(const std::string& key, int delta) { lineScore[key] += delta; }
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
  DynamicWeights dynamicWeights;
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
struct Formats {
  bool pgnEnabled = true;
  bool epdEnabled = true;
};

struct Integrity {
  bool antiCheatEnabled = false;
  bool checksumOk = true;
  bool verifyRuntime() const { return !antiCheatEnabled || checksumOk; }
};

struct TestHarness {
  bool regressionEnabled = true;
  bool eloEnabled = true;
  bool selfPlayTournaments = true;
  std::unordered_map<std::string, double> params;
};
}  // namespace tooling

}  // namespace engine_components

#endif
