#include <fstream>
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
  engine_components::eval_model::PolicyNet policy;
  engine_components::eval_model::TrainingInfra training;

  engine_components::opening::Book book;
  engine_components::opening::PrepModule prep;
  engine_components::timing::Manager timeManager;
  engine_components::tooling::Formats formats;
  engine_components::tooling::Integrity integrity;
  engine_components::tooling::TestHarness tests;
};

void log(State& state, const std::string& msg) {
  if (state.logFile.is_open()) {
    state.logFile << msg << '\n';
  }
}

std::string openingKey(const State& state) {
  if (state.board.history.empty()) {
    return "startpos";
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
  state.stopRequested = false;
  state.features.multiPV = 1;
  state.parallel.threads = 1;
  state.mcts.enabled = false;
  state.policy.enabled = true;
  state.policy.priors = {0.70f, 0.20f, 0.10f};
  state.cache.load(state.openingCachePath);
  state.logFile.open("engine.log", std::ios::app);
  log(state, "engine initialized with search/eval/tooling scaffolding");
}

void printUciId() {
  std::cout << "id name GameChessEngineX\n";
  std::cout << "id author Codex\n";
  std::cout << "option name Hash type spin default 64 min 1 max 8192\n";
  std::cout << "option name Threads type spin default 1 min 1 max 256\n";
  std::cout << "option name MultiPV type spin default 1 min 1 max 32\n";
  std::cout << "option name UseNNUE type check default false\n";
  std::cout << "option name UseMCTS type check default false\n";
  std::cout << "option name UseMultiRateThinking type check default true\n";
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
  } else if (name == "MultiPV") {
    state.features.multiPV = std::max(1, std::stoi(value));
  } else if (name == "UseNNUE") {
    state.nnue.enabled = (value == "true");
  } else if (name == "UseMCTS") {
    state.mcts.enabled = (value == "true");
    state.features.useMCTS = state.mcts.enabled;
  } else if (name == "UseMultiRateThinking") {
    state.features.useMultiRateThinking = (value == "true");
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
  return limits;
}

void handleGo(State& state, const std::string& cmd) {
  if (!state.integrity.verifyRuntime()) {
    std::cout << "info string integrity-check-failed\n";
    std::cout << "bestmove 0000\n";
    return;
  }

  const std::string key = openingKey(state);
  const std::string cached = state.cache.get(key);
  if (!cached.empty()) {
    std::cout << "info string cache_hit true\n";
    std::cout << "bestmove " << cached << '\n';
    return;
  }

  state.stopRequested = false;
  const search::Limits limits = parseGoLimits(state, cmd);

  search::Searcher searcher(state.features, &state.killer, &state.history, &state.counter, &state.pvTable, &state.see,
                            &state.handcrafted, &state.policy);
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
      state.perftNodes += movegen::generatePseudoLegal(state.board).size();
      std::cout << "info string perft_nodes " << state.perftNodes << '\n';
    } else if (input == "bench") {
      std::cout << "info string bench placeholders: search, nnue, mcts, movegen, tt\n";
    } else if (input == "buildbook") {
      std::cout << "info string book_lines " << state.prep.builder.lines.size() << '\n';
    } else if (input == "losslearn") {
      state.training.lossLearning.recordLoss();
      state.training.lossLearning.runAdversarialSweep();
      std::cout << "info string loss_learning cases=" << state.training.lossLearning.lossCases
                << " adversarial=" << state.training.lossLearning.adversarialTests << '\n';
    } else if (input == "integrity") {
      std::cout << "info string integrity " << (state.integrity.verifyRuntime() ? "ok" : "failed") << '\n';
    } else if (input == "explain") {
      std::cout << "info string explain " << state.handcrafted.breakdown() << '\n';
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
