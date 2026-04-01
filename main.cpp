#include <algorithm>
#include <array>
#include <cctype>
#include <cstdio>
#include <fstream>
#include <filesystem>
#include <numeric>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "board.h"
#include "engine_components.h"
#include "eval.h"
#include "movegen.h"
#include "search.h"
#include "tt.h"

namespace engine {

struct State {
  board::Board board;
  tt::Table tt;
  eval::Params evalParams;
  std::mt19937 rng{std::random_device{}()};
  std::ofstream logFile;
  bool running = true;
  bool stopRequested = false;
  bool adaptiveSwitching = true;
  std::uint64_t perftNodes = 0;
  std::string openingCachePath = "opening_cache.txt";

  engine_components::representation::AttackTables attacks;
  engine_components::representation::MagicTables magic;
  engine_components::hashing::Zobrist zobrist;
  engine_components::hashing::RepetitionTracker repetition;

  engine_components::search_arch::Features features;
  engine_components::search_arch::ParallelConfig parallel;
  engine_components::search_arch::MCTSConfig mcts;
  engine_components::search_helpers::KillerTable killer;
  engine_components::search_helpers::HistoryHeuristic history;
  engine_components::search_helpers::CounterMoveTable counter;
  engine_components::search_helpers::PVTable pvTable;
  engine_components::search_helpers::SEE see;
  engine_components::search_helpers::SearchResultCache cache;

  engine_components::eval_model::Handcrafted handcrafted;
  engine_components::eval_model::EndgameHeuristics endgame;
  engine_components::eval_model::NNUE nnue;
  engine_components::eval_model::StrategyNet strategyNet;
  engine_components::eval_model::PolicyNet policy;
  engine_components::eval_model::TransformerCritic transformerCritic;
  engine_components::eval_model::TrainingInfra training;

  engine_components::opening::Book book;
  engine_components::opening::PrepModule prep;
  engine_components::timing::Manager timeManager;
  engine_components::tooling::Formats formats;
  engine_components::tooling::Integrity integrity;
  engine_components::tooling::SyzygyConfig syzygy;
  engine_components::tooling::RamTablebase ramTablebase;
  engine_components::tooling::TestHarness tests;
};

void log(State& state, const std::string& msg) {
  if (state.logFile.is_open()) {
    state.logFile << msg << '\n';
  }
}

std::string describeFeatures(const State& state) {
  std::ostringstream out;
  out << "search[custom_ugds_only=" << state.features.useUGDS
      << " policyPrune=" << state.features.usePolicyPruning
      << " pvPrune=" << state.features.usePolicyValuePruning
      << " lazy=" << state.features.useLazyEval
      << " topK=" << state.features.policyTopK
      << " masterTop=" << state.features.masterEvalTopMoves << "] ";

  out << "nnue[enabled=" << state.nnue.enabled << " amx=" << state.nnue.cfg.useAMXPath
      << " inputs=" << state.nnue.cfg.inputs << " h1=" << state.nnue.cfg.hidden1
      << " h2=" << state.nnue.cfg.hidden2 << "] ";

  out << "meganet[enabled=" << state.strategyNet.enabled
      << " policyOut=" << state.strategyNet.cfg.policyOutputs
      << " hardPhase=" << state.strategyNet.cfg.useHardPhaseSwitch
      << " experts=" << state.strategyNet.cfg.activeExperts << "] ";

  out << "blitz[enabled=" << state.policy.enabled
      << " params=" << state.policy.parameterCount() << "] ";

  out << "transformer[enabled=" << state.transformerCritic.enabled
      << " params=" << state.transformerCritic.parameterCount() << "] ";

  out << "m2cts[batch=" << state.mcts.miniBatchSize
      << " vloss=" << state.mcts.virtualLoss
      << " phaseAware=" << state.mcts.usePhaseAwareM2CTS
      << " clade=" << state.mcts.useCladeSelection
      << " fpuRed=" << state.mcts.fpuReduction << "] ";

  out << "parallel[on=" << state.features.useParallel
      << " threads=" << state.parallel.threads
      << " ybwc=" << state.parallel.ybwcFirstMoveSerial
      << " splitDepth=" << state.parallel.splitDepthLimit
      << " splitMoves=" << state.parallel.maxSplitMoves
      << " deterministic=" << state.parallel.deterministicMode << "] ";

  out << "tooling[ramTB=" << state.ramTablebase.enabled
      << " ipcPath=" << state.tests.ipc.path
      << " binpack=" << state.tests.binpack.path
      << " cat=" << state.training.cat.enabled
      << " reviewer=" << state.training.reviewerEnabled
      << " topics=" << state.training.practiceTopics.size() << "]";

  return out.str();
}

std::string joinTopics(const std::vector<std::string>& topics) {
  if (topics.empty()) return "none";
  std::ostringstream oss;
  for (std::size_t i = 0; i < topics.size(); ++i) {
    if (i) oss << ',';
    oss << topics[i];
  }
  return oss.str();
}

bool startsWith(const std::string& text, const std::string& prefix) {
  return text.size() >= prefix.size() && text.compare(0, prefix.size(), prefix) == 0;
}

constexpr const char* kDefaultStrategyWeights = "meganet.lc0";

std::string resolveDefaultNNUEWeightsPath() {
  namespace fs = std::filesystem;
  constexpr const char* kPreferred = "eval.nnue";

  std::error_code ec;
  if (fs::is_regular_file(kPreferred, ec)) return kPreferred;

  const fs::path cwd = fs::current_path(ec);
  if (ec) return kPreferred;

  std::string fallback;
  for (const auto& entry : fs::directory_iterator(cwd, ec)) {
    if (ec || !entry.is_regular_file()) continue;

    const fs::path candidate = entry.path().filename();
    if (candidate.extension() != ".nnue") continue;

    const std::string name = candidate.string();
    if (startsWith(name, "nn-")) return name;
    if (fallback.empty()) fallback = name;
  }
  return fallback.empty() ? kPreferred : fallback;
}

std::string openingKey(const State& state) {
  if (state.board.history.empty()) {
    board::Board start;
    start.setStartPos();
    if (state.board.whiteToMove == start.whiteToMove && state.board.squares == start.squares) {
      return "startpos";
    }
    std::ostringstream oss;
    oss << (state.board.whiteToMove ? 'w' : 'b') << ':';
    for (char sq : state.board.squares) oss << sq;
    return oss.str();
  }
  std::ostringstream oss;
  for (std::size_t i = 0; i < state.board.history.size(); ++i) {
    if (i) oss << '_';
    oss << state.board.history[i];
  }
  return oss.str();
}

struct MaterialProfile {
  int pieceCount = 0;
  int whiteNonKing = 0;
  int blackNonKing = 0;
};

MaterialProfile materialProfile(const board::Board& board) {
  MaterialProfile profile;
  for (char piece : board.squares) {
    if (piece == '.') continue;
    ++profile.pieceCount;
    const char p = static_cast<char>(std::tolower(static_cast<unsigned char>(piece)));
    if (p == 'k') continue;
    if (std::isupper(static_cast<unsigned char>(piece))) ++profile.whiteNonKing;
    else ++profile.blackNonKing;
  }
  return profile;
}

std::string materialKey(const MaterialProfile& profile) {
  return "K" + std::to_string(profile.whiteNonKing) + "v" + std::to_string(profile.blackNonKing);
}

bool resolveTablebaseRootMove(State& state, movegen::Move& bestMove, int& rootWdl, std::string& rootKey) {
  if (!state.ramTablebase.enabled || !state.ramTablebase.loaded) return false;
  const auto legal = movegen::generateLegal(state.board);
  if (legal.empty()) return false;

  int bestScore = -2;
  bool found = false;
  for (const auto& move : legal) {
    board::Undo undo;
    if (!state.board.makeMove(move.from, move.to, move.promotion, undo)) continue;
    const MaterialProfile childProfile = materialProfile(state.board);
    const std::string childKey = materialKey(childProfile);
    const int childWdl = state.ramTablebase.probe(childKey);
    state.board.unmakeMove(move.from, move.to, move.promotion, undo);

    if (childWdl == 0) continue;
    const int score = -childWdl;
    if (!found || score > bestScore) {
      found = true;
      bestScore = score;
      bestMove = move;
      rootWdl = score;
      rootKey = childKey;
    }
  }
  return found;
}

std::string boardToFEN(const board::Board& b) {
  std::ostringstream oss;
  for (int rank = 7; rank >= 0; --rank) {
    int empties = 0;
    for (int file = 0; file < 8; ++file) {
      const char piece = b.squares[rank * 8 + file];
      if (piece == '.') {
        ++empties;
        continue;
      }
      if (empties > 0) {
        oss << empties;
        empties = 0;
      }
      oss << piece;
    }
    if (empties > 0) oss << empties;
    if (rank) oss << '/';
  }
  oss << ' ' << (b.whiteToMove ? 'w' : 'b') << ' ';
  std::string castling = "-";
  if (b.castlingRights) {
    castling.clear();
    if (b.castlingRights & 1) castling.push_back('K');
    if (b.castlingRights & 2) castling.push_back('Q');
    if (b.castlingRights & 4) castling.push_back('k');
    if (b.castlingRights & 8) castling.push_back('q');
  }
  oss << castling << ' ';
  if (b.enPassantSquare >= 0) oss << board::Board::squareName(b.enPassantSquare);
  else oss << '-';
  oss << ' ' << b.halfmoveClock << ' ' << b.fullmoveNumber;
  return oss.str();
}

std::string shellSingleQuote(const std::string& text) {
  std::string out = "'";
  for (char c : text) {
    if (c == '\'') out += "'\"'\"'";
    else out.push_back(c);
  }
  out.push_back('\'');
  return out;
}

struct SyzygyProbeResult {
  int wdl = 0;
  int dtz = 0;
  std::string bestMove;
};

bool probeSyzygy(const State& state, const search::Limits& limits, SyzygyProbeResult& out) {
  if (!state.syzygy.enabled || state.syzygy.probeCommand.empty()) return false;
  const MaterialProfile profile = materialProfile(state.board);
  if (profile.pieceCount > state.syzygy.probeLimit) return false;
  if (limits.depth > 0 && limits.depth < state.syzygy.probeDepth) return false;

  const std::string fen = boardToFEN(state.board);
  const std::string cmd = state.syzygy.probeCommand + " " + shellSingleQuote(fen);
  std::array<char, 256> buffer{};
  std::string output;
  FILE* pipe = popen(cmd.c_str(), "r");
  if (!pipe) return false;
  while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe) != nullptr) {
    output += buffer.data();
  }
  const int rc = pclose(pipe);
  if (rc != 0 || output.empty()) return false;

  std::istringstream iss(output);
  if (!(iss >> out.wdl)) return false;
  if (!(iss >> out.dtz)) out.dtz = 0;
  if (!(iss >> out.bestMove)) out.bestMove.clear();
  return true;
}

bool isTacticalPosition(const State& state) {
  if (state.board.inCheck(state.board.whiteToMove)) return true;
  const auto legal = movegen::generateLegal(state.board);
  int captures = 0;
  for (const auto& move : legal) {
    if (state.board.squares[move.to] != '.') ++captures;
  }
  return captures >= 3;
}

std::string encodeSearchBinpackRecord(const State& state, const std::string& key, const search::Result& result,
                                      const movegen::Move& bestMove, bool novel) {
  std::ostringstream record;
  const int valueStmCp = result.scoreCp;
  const int valueWhiteCp = state.board.whiteToMove ? result.scoreCp : -result.scoreCp;
  record << "BINPACK1";
  record << "|key=" << key;
  record << "|stm=" << (state.board.whiteToMove ? 'w' : 'b');
  record << "|board=";
  for (char sq : state.board.squares) record << sq;
  record << "|depth=" << result.depth;
  record << "|score_cp=" << result.scoreCp;
  record << "|value_stm_cp=" << valueStmCp;
  record << "|value_white_cp=" << valueWhiteCp;
  record << "|nodes=" << result.nodes;
  record << "|bestmove=" << bestMove.toUCI();
  record << "|novelty=" << (novel ? 1 : 0);
  record << "|pv=";
  for (std::size_t i = 0; i < result.pv.size(); ++i) {
    if (i) record << ',';
    record << result.pv[i].toUCI();
  }
  record << "|candidate_depths=";
  for (std::size_t i = 0; i < result.candidateDepths.size(); ++i) {
    if (i) record << ',';
    record << result.candidateDepths[i];
  }
  std::string thinking = result.evalBreakdown;
  std::replace(thinking.begin(), thinking.end(), '|', '/');
  record << "|thinking=" << thinking;
  return record.str();
}

void initialize(State& state) {
  state.board.setStartPos();
  state.tt.initialize(64);
  eval::initialize(state.evalParams);
  state.attacks.initialize();
  state.magic.initialize();
  state.zobrist.initialize();
  state.repetition.clear();
  state.repetition.push(tt::hash(state.board));
  state.perftNodes = 0;

  const MaterialProfile initProfile = materialProfile(state.board);
  if (initProfile.pieceCount <= 6) {
    const std::string tbKey = materialKey(initProfile);
    const int tbWdl = state.ramTablebase.probe(tbKey);
    if (tbWdl != 0) {
      std::cout << "info string ram_tablebase hit key=" << tbKey << " wdl=" << tbWdl << '\n';
    }
  }

  state.stopRequested = false;
  state.features.multiPV = 1;
  state.features.useUGDS = true;
  state.features.usePVS = false;
  state.features.useAspiration = false;
  state.features.useQuiescence = false;
  state.features.useNullMove = false;
  state.features.useLMR = false;
  state.features.useFutility = false;
  state.features.useMateDistancePruning = false;
  state.features.useExtensions = false;
  state.features.useMCTS = false;
  state.features.useParallel = false;
  state.features.useAsync = false;
  state.parallel.threads = 1;
  state.mcts.enabled = false;
  state.policy.enabled = true;
  state.features.usePolicyPruning = true;
  state.features.policyTopK = state.strategyNet.cfg.topKForPruning;
  state.features.policyPruneThreshold = state.strategyNet.cfg.pruneThreshold;
  state.features.useLazyEval = true;
  state.features.masterEvalTopMoves = 3;
  state.nnue.load(resolveDefaultNNUEWeightsPath());
  state.strategyNet.load(kDefaultStrategyWeights);
  state.transformerCritic.load("chess_transformer_25m.pt");
  state.policy.priors = {0.70f, 0.20f, 0.10f};
  state.policy.name = "blitz net";
  state.ramTablebase.enabled = false;
  state.cache.load(state.openingCachePath);
  state.logFile.open("engine.log", std::ios::app);
  log(state, "engine initialized with search/eval/tooling scaffolding");
  log(state, "nnue_params=" + std::to_string(state.nnue.parameterCount()) +
            " strategy_params=" + std::to_string(state.strategyNet.parameterCount()));
}

void printUciId() {
  std::cout << "id name GameChessEngineX\n";
  std::cout << "id author Codex\n";
  std::cout << "option name Hash type spin default 64 min 1 max 8192\n";
  std::cout << "option name MultiPV type spin default 1 min 1 max 32\n";
  std::cout << "option name UseNNUE type check default true\n";
  std::cout << "option name UseTransformerCritic type check default true\n";
  std::cout << "option name EnableCAT type check default true\n";
  std::cout << "option name UseStrategyNN type check default true\n";
  std::cout << "option name StrategyPolicyOutputs type spin default 4096 min 64 max 4096\n";
  std::cout << "option name UseMultiRateThinking type check default true\n";
  std::cout << "option name EnableDistillation type check default false\n";
  std::cout << "option name UsePolicyPruning type check default true\n";
  std::cout << "option name PolicyTopK type spin default 5 min 1 max 32\n";
  std::cout << "option name UseLazyEval type check default true\n";
  std::cout << "option name MasterEvalTopMoves type spin default 3 min 1 max 8\n";
  std::cout << "option name UseAMXNNUEPath type check default false\n";
  std::cout << "option name StrategyUseHardPhaseSwitch type check default true\n";
  std::cout << "option name StrategyActiveExperts type spin default 2 min 1 max 2\n";
  std::cout << "option name UseAdaptiveSwitching type check default true\n";
  std::cout << "option name UseSyzygy type check default false\n";
  std::cout << "option name SyzygyProbeLimit type spin default 7 min 3 max 7\n";
  std::cout << "option name SyzygyProbeDepth type spin default 1 min 0 max 64\n";
  std::cout << "option name SyzygyProbeCommand type string default \n";
  std::cout << "option name UseRamTablebase type check default false\n";
  std::cout << "option name AntiCheat type check default false\n";
  std::cout << "option name ReviewerEnabled type check default false\n";
  std::cout << "option name ReviewerWeights type string default reviewer.nn\n";
  std::cout << "uciok\n";
}

void handleSetOption(State& state, const std::string& cmd) {
  std::istringstream iss(cmd);
  std::string token;
  std::string name;
  std::string value;
  iss >> token;
  while (iss >> token) {
    if (token == "name") {
      while (iss >> token && token != "value") {
        if (!name.empty()) name += " ";
        name += token;
      }
      if (token != "value") break;
      std::getline(iss, value);
      if (!value.empty() && value.front() == ' ') value.erase(0, 1);
      break;
    }
  }

  if (name == "Hash") {
    int mb = std::max(1, std::stoi(value));
    state.tt.initialize(static_cast<std::size_t>(mb));
  } else if (name == "MultiPV") {
    state.features.multiPV = std::max(1, std::stoi(value));
  } else if (name == "UseNNUE") {
    state.nnue.enabled = (value == "true");
  } else if (name == "UseTransformerCritic") {
    state.transformerCritic.enabled = (value == "true");
  } else if (name == "EnableCAT") {
    state.training.cat.enabled = (value == "true");
  } else if (name == "UseStrategyNN") {
    state.strategyNet.enabled = (value == "true");
  } else if (name == "UseBook") {
    state.book.enabled = (value == "true");
  } else if (name == "StrategyPolicyOutputs") {
    state.strategyNet.cfg.policyOutputs = std::max(64, std::stoi(value));
    state.strategyNet.load(state.strategyNet.weightsPath);
  } else if (name == "UseMultiRateThinking") {
    state.features.useMultiRateThinking = (value == "true");
  } else if (name == "EnableDistillation") {
    state.training.distillationEnabled = (value == "true");
  } else if (name == "UsePolicyPruning") {
    state.features.usePolicyPruning = (value == "true");
  } else if (name == "PolicyTopK") {
    state.features.policyTopK = std::max(1, std::stoi(value));
  } else if (name == "UseLazyEval") {
    state.features.useLazyEval = (value == "true");
  } else if (name == "MasterEvalTopMoves") {
    state.features.masterEvalTopMoves = std::clamp(std::stoi(value), 1, 8);
  } else if (name == "UseAMXNNUEPath") {
    state.nnue.cfg.useAMXPath = (value == "true");
  } else if (name == "StrategyUseHardPhaseSwitch") {
    state.strategyNet.cfg.useHardPhaseSwitch = (value == "true");
  } else if (name == "StrategyActiveExperts") {
    state.strategyNet.cfg.activeExperts = std::clamp(std::stoi(value), 1, 2);
  } else if (name == "UseAdaptiveSwitching") {
    state.adaptiveSwitching = (value == "true");
  } else if (name == "UseSyzygy") {
    state.syzygy.enabled = (value == "true");
  } else if (name == "SyzygyProbeLimit") {
    state.syzygy.probeLimit = std::clamp(std::stoi(value), 3, 7);
  } else if (name == "SyzygyProbeDepth") {
    state.syzygy.probeDepth = std::max(0, std::stoi(value));
  } else if (name == "SyzygyProbeCommand") {
    state.syzygy.probeCommand = value;
  } else if (name == "ReviewerEnabled") {
    state.training.reviewerEnabled = (value == "true");
  } else if (name == "ReviewerWeights") {
    state.training.registerReviewNetwork(value);
  } else if (name == "UseRamTablebase") {
    state.ramTablebase.enabled = (value == "true");
    if (state.ramTablebase.enabled && !state.ramTablebase.loaded) state.ramTablebase.preload6ManMock();
  } else if (name == "AntiCheat") {
    state.integrity.antiCheatEnabled = (value == "true");
  }
}

void handlePosition(State& state, const std::string& cmd) {
  std::istringstream iss(cmd);
  std::string token;
  iss >> token;
  if (!(iss >> token)) return;

  if (token == "startpos") {
    state.board.setStartPos();
  } else if (token == "fen") {
    std::vector<std::string> fenParts;
    for (int i = 0; i < 6 && (iss >> token); ++i) {
      if (token == "moves") break;
      fenParts.push_back(token);
    }
    std::ostringstream fen;
    for (std::size_t i = 0; i < fenParts.size(); ++i) {
      if (i) fen << ' ';
      fen << fenParts[i];
    }
    if (!state.board.setFromFEN(fen.str())) {
      std::cout << "info string invalid fen\n";
      return;
    }
    if (token != "moves") return;
  }
  state.repetition.clear();
  state.repetition.push(tt::hash(state.board));

  if (token != "moves") {
    if (!(iss >> token) || token != "moves") return;
  }

  while (iss >> token) {
    movegen::Move mv;
    if (!movegen::parseUCIMove(token, mv) || !movegen::isLegalMove(state.board, mv)) {
      std::cout << "info string illegal move " << token << '\n';
      log(state, "illegal move: " + token);
      continue;
    }
    if (state.board.applyMove(mv.from, mv.to, mv.promotion)) {
      state.board.history.push_back(token);
      state.repetition.push(tt::hash(state.board));
    }
  }
}

search::Limits parseGoLimits(State& state, const std::string& cmd) {
  search::Limits limits;
  std::istringstream iss(cmd);
  std::string token;
  iss >> token;
  while (iss >> token) {
    if (token == "depth") {
      iss >> limits.depth;
    } else if (token == "movetime") {
      iss >> limits.movetimeMs;
    } else if (token == "infinite") {
      limits.infinite = true;
    } else if (token == "wtime" || token == "btime") {
      iss >> state.timeManager.remainingMs;
    } else if (token == "winc" || token == "binc") {
      iss >> state.timeManager.incrementMs;
    }
  }
  if (limits.movetimeMs == 0 && state.timeManager.remainingMs > 0) {
    limits.movetimeMs = state.timeManager.allocateMoveTimeMs(25);
  }
  if (state.features.useParallel && state.parallel.threads > 1) {
    const int overhead = std::max(1, state.parallel.threads / 2);
    limits.movetimeMs = std::max(1, limits.movetimeMs - overhead);
  }
  return limits;
}

void tuneMoveTimeForPosition(State& state, const MaterialProfile& profile, bool tactical, search::Limits& limits) {
  constexpr int kLowLegalCount = 8;
  constexpr int kHighLegalCount = 35;
  constexpr int kEndgamePieceCount = 10;
  constexpr int kTacticalNum = 5, kTacticalDen = 4;      // 1.25x
  constexpr int kLowLegalNum = 3, kLowLegalDen = 2;      // 1.50x
  constexpr int kHighLegalNum = 17, kHighLegalDen = 20;  // 0.85x
  constexpr int kEndgameNum = 7, kEndgameDen = 5;        // 1.40x
  constexpr int kRemainingCapDivisor = 5;
  constexpr int kIncrementCapMultiplier = 2;

  if (limits.movetimeMs <= 0 || limits.infinite) return;

  int tuned = limits.movetimeMs;
  const int legalCount = static_cast<int>(movegen::generateLegal(state.board).size());

  if (tactical) tuned = tuned * kTacticalNum / kTacticalDen;
  if (legalCount <= kLowLegalCount) tuned = tuned * kLowLegalNum / kLowLegalDen;
  else if (legalCount >= kHighLegalCount) tuned = tuned * kHighLegalNum / kHighLegalDen;
  if (profile.pieceCount <= kEndgamePieceCount) tuned = tuned * kEndgameNum / kEndgameDen;

  if (state.timeManager.remainingMs > 0) {
    const int cap = std::max(1, state.timeManager.remainingMs / kRemainingCapDivisor +
                                    state.timeManager.incrementMs * kIncrementCapMultiplier);
    tuned = std::min(tuned, cap);
  }
  limits.movetimeMs = std::max(1, tuned);
}

void handleGo(State& state, const std::string& cmd) {
  if (!state.integrity.verifyRuntime()) {
    std::cout << "info string integrity-check-failed\n";
    std::cout << "bestmove 0000\n";
    return;
  }

  if (state.board.halfmoveClock >= 100) {
    std::cout << "info string draw by rule\n";
    std::cout << "bestmove 0000\n";
    return;
  }
  if (state.repetition.isThreefold(tt::hash(state.board))) {
    std::cout << "info string draw by repetition\n";
    std::cout << "bestmove 0000\n";
    return;
  }

  const std::string key = openingKey(state);
  const std::string bookMove = state.book.probe(key);
  if (!bookMove.empty()) {
    std::cout << "info string book_hit true\n";
    std::cout << "bestmove " << bookMove << "\n";
    return;
  }
  const std::string cached = state.cache.get(key);
  if (!cached.empty()) {
    std::cout << "info string cache_hit true\n";
    std::cout << "bestmove " << cached << '\n';
    return;
  }

  const MaterialProfile profile = materialProfile(state.board);
  if (profile.pieceCount <= 6) {
    const std::string tbKey = materialKey(profile);
    const int tbWdl = state.ramTablebase.probe(tbKey);
    if (tbWdl != 0) {
      std::cout << "info string ram_tablebase hit key=" << tbKey << " wdl=" << tbWdl << '\n';
      movegen::Move tbMove;
      int rootWdl = 0;
      std::string childKey;
      if (resolveTablebaseRootMove(state, tbMove, rootWdl, childKey)) {
        std::cout << "info string ram_tablebase exact_root true child_key=" << childKey
                  << " root_wdl=" << rootWdl << '\n';
        std::cout << "bestmove " << tbMove.toUCI() << "\n";
        return;
      }
    }
  }

  state.stopRequested = false;
  search::Limits limits = parseGoLimits(state, cmd);
  const bool tactical = isTacticalPosition(state);
  tuneMoveTimeForPosition(state, profile, tactical, limits);
  if (state.syzygy.enabled) {
    SyzygyProbeResult syz;
    if (probeSyzygy(state, limits, syz)) {
      std::cout << "info string syzygy_hit true wdl=" << syz.wdl << " dtz=" << syz.dtz << '\n';
      movegen::Move parsed;
      if (!syz.bestMove.empty() && movegen::parseUCIMove(syz.bestMove, parsed) && movegen::isLegalMove(state.board, parsed)) {
        std::cout << "bestmove " << syz.bestMove << "\n";
        return;
      }
    }
  }
  constexpr int kForceAlphaBetaPieceCount = 12;
  const bool forceAlphaBeta = state.adaptiveSwitching && (profile.pieceCount <= kForceAlphaBetaPieceCount || tactical);
  const bool allowMcts = state.features.useMCTS && !forceAlphaBeta;

  auto searchFeatures = state.features;
  searchFeatures.useMCTS = allowMcts;

  search::Searcher searcher(searchFeatures, &state.killer, &state.history, &state.counter, &state.pvTable, &state.see,
                            &state.handcrafted, &state.policy, &state.nnue, &state.strategyNet, &state.transformerCritic,
                            state.mcts, state.parallel, &state.tt);
  const search::Result result = searcher.think(state.board, limits, state.rng, &state.stopRequested);
  movegen::Move safeBest = result.bestMove;
  if (!movegen::isLegalMove(state.board, safeBest)) {
    const auto legalFallback = movegen::generateLegal(state.board);
    if (!legalFallback.empty()) {
      safeBest = legalFallback.front();
      std::cout << "info string repaired_illegal_bestmove true\n";
    }
  }

  bool novel = state.prep.novelty.isNovel(key);
  std::cout << "info depth " << result.depth << " nodes " << result.nodes << " score cp " << result.scoreCp << " pv";
  for (const auto& move : result.pv) {
    std::cout << ' ' << move.toUCI();
  }
  std::cout << " info string novelty=" << (novel ? "true" : "false") << '\n';

  if (!result.candidateDepths.empty()) {
    std::cout << "info string multi_rate_depths";
    for (int d : result.candidateDepths) std::cout << ' ' << d;
    std::cout << '\n';
  }

  std::cout << "info string eval_breakdown " << result.evalBreakdown << '\n';

  std::cout << "bestmove " << safeBest.toUCI();
  if (result.ponder.from >= 0) {
    std::cout << " ponder " << result.ponder.toUCI();
  }
  std::cout << '\n';

  const std::string binpackRecord = encodeSearchBinpackRecord(state, key, result, safeBest, novel);
  const bool wroteBinpack = state.tests.binpack.appendRecord(binpackRecord);
  std::cout << "info string binpack_append " << (wroteBinpack ? "ok" : "failed") << '\n';

  state.cache.put(key, safeBest.toUCI());
  state.prep.builder.addLine(key, safeBest.toUCI());
}

std::uint64_t perft(const board::Board& position, int depth) {
  if (depth <= 0) return 1;
  const auto moves = movegen::generateLegal(position);
  if (depth == 1) return static_cast<std::uint64_t>(moves.size());

  std::uint64_t nodes = 0;
  for (const auto& mv : moves) {
    board::Board child = position;
    if (!child.applyMove(mv.from, mv.to, mv.promotion)) continue;
    nodes += perft(child, depth - 1);
  }
  return nodes;
}

void runLoop(State& state) {
  std::string input;
  while (state.running && std::getline(std::cin, input)) {
    if (input == "uci") {
      printUciId();
    } else if (input == "isready") {
      std::cout << "readyok\n";
    } else if (input.rfind("setoption", 0) == 0) {
      handleSetOption(state, input);
    } else if (input.rfind("position", 0) == 0) {
      handlePosition(state, input);
    } else if (input.rfind("go", 0) == 0) {
      handleGo(state, input);
    } else if (input == "stop") {
      state.stopRequested = true;
    } else if (input.rfind("perft", 0) == 0) {
      std::istringstream iss(input);
      std::string token;
      int depth = 1;
      iss >> token;
      if (!(iss >> depth)) depth = 1;
      depth = std::max(0, depth);
      const std::uint64_t nodes = perft(state.board, depth);
      state.perftNodes += nodes;
      std::cout << "nodes " << nodes << '\n';
      std::cout << "info string perft_accumulated " << state.perftNodes << '\n';
    } else if (input == "bench") {
      const auto legal = movegen::generateLegal(state.board).size();
      std::cout << "info string bench movegen_legal=" << legal
                << " nnue_params=" << state.nnue.parameterCount()
                << " strategy_params=" << state.strategyNet.parameterCount()
                << " tt_entries=" << state.tt.entries.size()
                << " mcts_batch=" << state.mcts.miniBatchSize << "\n";
    } else if (input == "buildbook") {
      int imported = 0;
      for (const auto& kv : state.prep.builder.lines) {
        if (kv.first.empty() || kv.second.empty()) continue;
        state.book.moveByKey[kv.first] = kv.second;
        ++imported;
      }
      if (imported > 0) state.book.enabled = true;
      std::cout << "info string book_lines " << state.prep.builder.lines.size()
                << " imported=" << imported
                << " entries=" << state.book.moveByKey.size()
                << " enabled=" << (state.book.enabled ? "true" : "false") << '\n';
    } else if (input == "ipcmetrics") {
      engine_components::tooling::TrainingMetrics m;
      m.currentLoss = static_cast<float>(state.training.lossLearning.lossCases);
      m.eloGain = static_cast<float>(state.training.lossLearning.adversarialTests) * 0.1f;
      m.nodesPerSecond = 1000000;
      const std::string msg = "training-active";
      std::memcpy(m.statusMsg.data(), msg.c_str(), std::min(msg.size(), m.statusMsg.size() - 1));
      const bool ok = state.tests.ipc.write(m);
      std::cout << "info string ipc_metrics " << (ok ? "written" : "write_failed") << '\n';
    } else if (input == "binpackstats") {
      const std::size_t positions = state.tests.binpack.estimatePositionThroughput();
      std::cout << "info string binpack_positions_est " << positions << '\n';
    } else if (input == "losslearn") {
      state.training.lossLearning.recordLoss();
      state.training.lossLearning.runAdversarialSweep();
      if (state.training.distillationEnabled && state.strategyNet.enabled) {
        std::vector<float> planes(static_cast<std::size_t>(state.strategyNet.cfg.planes), 0.0f);
        for (int sq = 0; sq < 64; ++sq) {
          const char piece = state.board.squares[static_cast<std::size_t>(sq)];
          if (piece == '.') continue;
          const int idx = std::min(state.strategyNet.cfg.planes - 1, std::tolower(static_cast<unsigned char>(piece)) - 'a');
          if (idx >= 0 && idx < state.strategyNet.cfg.planes) planes[static_cast<std::size_t>(idx)] += 1.0f / 8.0f;
        }
        int nonPawnMaterial = 0;
        for (char piece : state.board.squares) {
          const char p = static_cast<char>(std::tolower(static_cast<unsigned char>(piece)));
          if (p == 'n' || p == 'b') nonPawnMaterial += 3;
          if (p == 'r') nonPawnMaterial += 5;
          if (p == 'q') nonPawnMaterial += 9;
        }
        const auto phase = nonPawnMaterial >= 36 ? engine_components::eval_model::GamePhase::Opening
                         : nonPawnMaterial >= 16 ? engine_components::eval_model::GamePhase::Middlegame
                                                 : engine_components::eval_model::GamePhase::Endgame;
        const auto strategy = state.strategyNet.evaluate(planes, phase);
        const float policySignal = strategy.policy.empty() ? 0.0f
            : std::accumulate(strategy.policy.begin(), strategy.policy.end(), 0.0f) / static_cast<float>(strategy.policy.size());
        state.nnue.distillStrategicHint(policySignal, static_cast<float>(strategy.valueCp) / 1000.0f);
      }
      std::cout << "info string loss_learning cases=" << state.training.lossLearning.lossCases
                << " adversarial=" << state.training.lossLearning.adversarialTests
                << " distill=" << (state.training.distillationEnabled ? "on" : "off") << '\n';
    } else if (input == "integrity") {
      std::cout << "info string integrity " << (state.integrity.verifyRuntime() ? "ok" : "failed") << '\n';
    } else if (input == "explain") {
      std::cout << "info string explain " << state.handcrafted.breakdown() << '\n';
    } else if (input == "features") {
      std::cout << "info string features " << describeFeatures(state) << '\n';
    } else if (input == "weights") {
      std::cout << "info string weights nnue=" << state.nnue.weightsPath
                << " strategy=" << state.strategyNet.weightsPath
                << " reviewer=" << state.training.reviewerWeightsPath
                << " reviewer_enabled=" << (state.training.reviewerEnabled ? "true" : "false") << '\n';
    } else if (startsWith(input, "reviewgame ")) {
      if (!state.training.reviewerEnabled) {
        std::cout << "info string reviewer disabled; enable via setoption name ReviewerEnabled value true\n";
      } else {
        const std::string gameRecord = input.substr(std::string("reviewgame ").size());
        const auto topics = state.training.reviewGameAndSuggestPracticeTopics(gameRecord);
        std::cout << "info string practice_topics " << joinTopics(topics)
                  << " reviewer_weights=" << state.training.reviewerWeightsPath << '\n';
      }
    } else if (input == "practicetopics") {
      std::cout << "info string practice_topics " << joinTopics(state.training.practiceTopics) << '\n';
    } else if (input == "quit") {
      state.running = false;
    } else if (!input.empty()) {
      std::cout << "info string unknown command: " << input << '\n';
    }
  }
}

void shutdown(State& state) {
  state.tt.clear();
  state.cache.save(state.openingCachePath);
  if (state.logFile.is_open()) {
    log(state, "engine shutdown");
    state.logFile.close();
  }
}

}  // namespace engine

int main() {
  engine::State state;
  engine::initialize(state);
  engine::runLoop(state);
  engine::shutdown(state);
  return 0;
}
