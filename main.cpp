#include <algorithm>
#include <cctype>
#include <fstream>
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
  engine_components::eval_model::TrainingInfra training;

  engine_components::opening::Book book;
  engine_components::opening::PrepModule prep;
  engine_components::timing::Manager timeManager;
  engine_components::tooling::Formats formats;
  engine_components::tooling::Integrity integrity;
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
  out << "search[pvs=" << state.features.usePVS << " aspiration=" << state.features.useAspiration
      << " null=" << state.features.useNullMove << " lmr=" << state.features.useLMR
      << " qsearch=" << state.features.useQuiescence << " mcts=" << state.features.useMCTS
      << " policyPrune=" << state.features.usePolicyPruning
      << " pvPrune=" << state.features.usePolicyValuePruning
      << " lazy=" << state.features.useLazyEval
      << " topK=" << state.features.policyTopK
      << " masterTop=" << state.features.masterEvalTopMoves << "] ";

  out << "nnue[enabled=" << state.nnue.enabled << " amx=" << state.nnue.cfg.useAMXPath
      << " inputs=" << state.nnue.cfg.inputs << " h1=" << state.nnue.cfg.hidden1
      << " h2=" << state.nnue.cfg.hidden2 << "] ";

  out << "strategy[enabled=" << state.strategyNet.enabled
      << " policyOut=" << state.strategyNet.cfg.policyOutputs
      << " hardPhase=" << state.strategyNet.cfg.useHardPhaseSwitch
      << " experts=" << state.strategyNet.cfg.activeExperts << "] ";

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
      << " cat=" << state.training.cat.enabled << "]";

  return out.str();
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

void initialize(State& state) {
  state.board.setStartPos();
  state.tt.initialize(64);
  eval::initialize(state.evalParams);
  state.attacks.initialize();
  state.magic.initialize();
  state.zobrist.initialize();
  state.repetition.clear();
  state.perftNodes = 0;

  int pieceCount = 0;
  int whiteNonKing = 0;
  int blackNonKing = 0;
  for (char piece : state.board.squares) {
    if (piece == '.') continue;
    ++pieceCount;
    const char p = static_cast<char>(std::tolower(static_cast<unsigned char>(piece)));
    if (p == 'k') continue;
    if (std::isupper(static_cast<unsigned char>(piece))) ++whiteNonKing;
    else ++blackNonKing;
  }
  if (pieceCount <= 6) {
    const std::string tbKey = "K" + std::to_string(whiteNonKing) + "v" + std::to_string(blackNonKing);
    const int tbWdl = state.ramTablebase.probe(tbKey);
    if (tbWdl != 0) {
      std::cout << "info string ram_tablebase hit key=" << tbKey << " wdl=" << tbWdl << '\n';
    }
  }

  state.stopRequested = false;
  state.features.multiPV = 1;
  state.parallel.threads = 1;
  state.mcts.enabled = false;
  state.policy.enabled = true;
  state.features.usePolicyPruning = true;
  state.features.policyTopK = state.strategyNet.cfg.topKForPruning;
  state.features.policyPruneThreshold = state.strategyNet.cfg.pruneThreshold;
  state.features.useLazyEval = true;
  state.features.masterEvalTopMoves = 3;
  state.nnue.load("nnue.bin");
  state.strategyNet.load("strategy_large.nn");
  state.policy.priors = {0.70f, 0.20f, 0.10f};
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
  std::cout << "option name Threads type spin default 1 min 1 max 256\n";
  std::cout << "option name UseParallelSearch type check default false\n";
  std::cout << "option name SplitDepthLimit type spin default 8 min 1 max 32\n";
  std::cout << "option name YBWCFirstMoveSerial type check default true\n";
  std::cout << "option name MaxSplitMoves type spin default 6 min 1 max 32\n";
  std::cout << "option name DeterministicMode type check default false\n";
  std::cout << "option name MultiPV type spin default 1 min 1 max 32\n";
  std::cout << "option name UseNNUE type check default true\n";
  std::cout << "option name UseMCTS type check default false\n";
  std::cout << "option name MCTSBatchSize type spin default 256 min 32 max 2048\n";
  std::cout << "option name MCTSVirtualLoss type string default 0.25\n";
  std::cout << "option name MCTSUsePhaseAware type check default true\n";
  std::cout << "option name MCTSUseCladeSelection type check default true\n";
  std::cout << "option name MCTSFpuReduction type string default 0.20\n";
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
  std::cout << "option name UseRamTablebase type check default false\n";
  std::cout << "option name AntiCheat type check default false\n";
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
  } else if (name == "Threads") {
    state.parallel.threads = std::max(1, std::stoi(value));
  } else if (name == "UseParallelSearch") {
    state.features.useParallel = (value == "true");
  } else if (name == "SplitDepthLimit") {
    state.parallel.splitDepthLimit = std::clamp(std::stoi(value), 1, 32);
  } else if (name == "YBWCFirstMoveSerial") {
    state.parallel.ybwcFirstMoveSerial = (value == "true");
  } else if (name == "MaxSplitMoves") {
    state.parallel.maxSplitMoves = std::clamp(std::stoi(value), 1, 32);
  } else if (name == "DeterministicMode") {
    state.parallel.deterministicMode = (value == "true");
  } else if (name == "MultiPV") {
    state.features.multiPV = std::max(1, std::stoi(value));
  } else if (name == "UseNNUE") {
    state.nnue.enabled = (value == "true");
  } else if (name == "UseMCTS") {
    state.mcts.enabled = (value == "true");
    state.features.useMCTS = state.mcts.enabled;
  } else if (name == "MCTSBatchSize") {
    state.mcts.miniBatchSize = std::clamp(std::stoi(value), 32, 2048);
  } else if (name == "MCTSVirtualLoss") {
    state.mcts.virtualLoss = std::clamp(std::stof(value), 0.0f, 2.0f);
  } else if (name == "MCTSUsePhaseAware") {
    state.mcts.usePhaseAwareM2CTS = (value == "true");
  } else if (name == "MCTSUseCladeSelection") {
    state.mcts.useCladeSelection = (value == "true");
  } else if (name == "MCTSFpuReduction") {
    state.mcts.fpuReduction = std::clamp(std::stof(value), 0.0f, 1.0f);
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
      state.repetition.push(static_cast<std::uint64_t>(state.board.history.size()));
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

void handleGo(State& state, const std::string& cmd) {
  if (!state.integrity.verifyRuntime()) {
    std::cout << "info string integrity-check-failed\n";
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

  int pieceCount = 0;
  int whiteNonKing = 0;
  int blackNonKing = 0;
  for (char piece : state.board.squares) {
    if (piece == '.') continue;
    ++pieceCount;
    const char p = static_cast<char>(std::tolower(static_cast<unsigned char>(piece)));
    if (p == 'k') continue;
    if (std::isupper(static_cast<unsigned char>(piece))) ++whiteNonKing;
    else ++blackNonKing;
  }
  if (pieceCount <= 6) {
    const std::string tbKey = "K" + std::to_string(whiteNonKing) + "v" + std::to_string(blackNonKing);
    const int tbWdl = state.ramTablebase.probe(tbKey);
    if (tbWdl != 0) {
      std::cout << "info string ram_tablebase hit key=" << tbKey << " wdl=" << tbWdl << '\n';
    }
  }

  state.stopRequested = false;
  const search::Limits limits = parseGoLimits(state, cmd);

  search::Searcher searcher(state.features, &state.killer, &state.history, &state.counter, &state.pvTable, &state.see,
                            &state.handcrafted, &state.policy, &state.nnue, &state.strategyNet, state.mcts, state.parallel, &state.tt);
  const search::Result result = searcher.think(state.board, limits, state.rng, &state.stopRequested);

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

  std::cout << "bestmove " << result.bestMove.toUCI();
  if (result.ponder.from >= 0) {
    std::cout << " ponder " << result.ponder.toUCI();
  }
  std::cout << '\n';

  state.cache.put(key, result.bestMove.toUCI());
  state.prep.builder.addLine(key, result.bestMove.toUCI());
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
    } else if (input == "perft") {
      state.perftNodes += movegen::generateLegal(state.board).size();
      std::cout << "info string perft_nodes " << state.perftNodes << '\n';
    } else if (input == "bench") {
      const auto pseudo = movegen::generatePseudoLegal(state.board).size();
      const auto legal = movegen::generateLegal(state.board).size();
      std::cout << "info string bench movegen_pseudo=" << pseudo
                << " movegen_legal=" << legal
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
