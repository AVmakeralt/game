#include <algorithm>
#include <chrono>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>

#include "board.h"
#include "eval.h"
#include "movegen.h"
#include "search.h"
#include "tt.h"

namespace {

struct EngineState {
  board::Board board;
  eval::Params evalParams;
  tt::Table table;
  bool running = true;
  bool stop = false;
  int hashMb = 64;
  std::unordered_map<std::uint64_t, int> repetition;
};

void trackPosition(EngineState& s) {
  ++s.repetition[tt::hash(s.board)];
}

void resetTracking(EngineState& s) {
  s.repetition.clear();
  trackPosition(s);
}

bool isDrawByRule(const EngineState& s) {
  if (s.board.halfmoveClock >= 100) return true;
  auto it = s.repetition.find(tt::hash(s.board));
  return it != s.repetition.end() && it->second >= 3;
}

std::uint64_t perft(board::Board& b, int depth) {
  if (depth == 0) return 1;
  std::uint64_t nodes = 0;
  auto moves = movegen::generateLegal(b);
  for (const auto& m : moves) {
    board::Undo u;
    if (!b.makeMove(m.from, m.to, m.promotion, u)) continue;
    nodes += perft(b, depth - 1);
    b.unmakeMove(m.from, m.to, m.promotion, u);
  }
  return nodes;
}

void printUci() {
  std::cout << "id name GameChessEngineX\n";
  std::cout << "id author Codex\n";
  std::cout << "option name Hash type spin default 64 min 1 max 8192\n";
  std::cout << "uciok\n";
}

void handlePosition(EngineState& s, const std::string& line) {
  std::istringstream iss(line);
  std::string tok;
  iss >> tok;
  iss >> tok;

  if (tok == "startpos") {
    s.board.setStartPos();
    resetTracking(s);
    if (!(iss >> tok)) return;
  } else if (tok == "fen") {
    std::string fen;
    std::string part;
    for (int i = 0; i < 6 && iss >> part; ++i) {
      if (part == "moves") {
        tok = "moves";
        break;
      }
      if (!fen.empty()) fen += ' ';
      fen += part;
    }
    if (!s.board.setFromFEN(fen)) {
      std::cout << "info string invalid fen\n";
      return;
    }
    resetTracking(s);
    if (tok != "moves" && !(iss >> tok)) return;
  }

  if (tok != "moves") return;
  while (iss >> tok) {
    movegen::Move m;
    if (!movegen::parseUCIMove(tok, m) || !movegen::isLegalMove(s.board, m)) {
      std::cout << "info string illegal move " << tok << "\n";
      continue;
    }
    board::Undo u;
    s.board.makeMove(m.from, m.to, m.promotion, u);
    s.board.history.push_back(tok);
    trackPosition(s);
  }
}

search::Limits parseGo(const std::string& line) {
  search::Limits l;
  std::istringstream iss(line);
  std::string tok;
  iss >> tok;
  while (iss >> tok) {
    if (tok == "depth") iss >> l.depth;
    else if (tok == "movetime") iss >> l.movetimeMs;
    else if (tok == "infinite") l.infinite = true;
  }
  return l;
}

void loop(EngineState& s) {
  std::string line;
  while (s.running && std::getline(std::cin, line)) {
    if (line == "uci")
      printUci();
    else if (line == "isready")
      std::cout << "readyok\n";
    else if (line.rfind("setoption", 0) == 0) {
      if (line.find("name Hash") != std::string::npos) {
        auto p = line.find("value ");
        if (p != std::string::npos) {
          s.hashMb = std::max(1, std::stoi(line.substr(p + 6)));
          s.table.initialize(static_cast<std::size_t>(s.hashMb));
        }
      }
    } else if (line.rfind("position", 0) == 0) {
      handlePosition(s, line);
    } else if (line.rfind("go", 0) == 0) {
      auto legalNow = movegen::generateLegal(s.board);
      if (legalNow.empty()) {
        if (s.board.inCheck(s.board.whiteToMove)) std::cout << "info string checkmate\n";
        else std::cout << "info string stalemate\n";
        std::cout << "bestmove 0000\n";
        continue;
      }
      if (isDrawByRule(s)) {
        std::cout << "info string draw by rule\n";
        std::cout << "bestmove 0000\n";
        continue;
      }
      auto l = parseGo(line);
      s.stop = false;
      s.table.newSearch();
      auto t0 = std::chrono::steady_clock::now();
      search::Searcher searcher(s.evalParams, &s.table, &s.stop);
      auto r = searcher.think(s.board, l);
      auto t1 = std::chrono::steady_clock::now();
      auto ms = std::max<long long>(1, std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count());
      std::uint64_t nps = static_cast<std::uint64_t>((r.nodes * 1000ULL) / static_cast<std::uint64_t>(ms));
      std::cout << "info depth " << r.depth << " nodes " << r.nodes << " time " << ms << " nps " << nps
                << " tthit " << s.table.hits() << '/' << s.table.probes() << " score cp " << r.scoreCp << " pv";
      for (const auto& m : r.pv) std::cout << ' ' << m.toUCI();
      std::cout << "\n";
      std::cout << "bestmove " << r.bestMove.toUCI() << "\n";
    } else if (line.rfind("perft", 0) == 0) {
      int depth = 1;
      std::istringstream iss(line);
      std::string cmd;
      iss >> cmd >> depth;
      auto t0 = std::chrono::steady_clock::now();
      auto nodes = perft(s.board, depth);
      auto t1 = std::chrono::steady_clock::now();
      auto ms = std::max<long long>(1, std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count());
      std::cout << "nodes " << nodes << " time " << ms << " nps " << (nodes * 1000ULL / static_cast<std::uint64_t>(ms)) << "\n";
    } else if (line == "stop") {
      s.stop = true;
    } else if (line == "quit") {
      s.running = false;
    }
  }
}

}  // namespace

int main() {
  EngineState state;
  state.board.setStartPos();
  eval::initialize(state.evalParams);
  tt::initializeZobrist();
  state.table.initialize(64);
  resetTracking(state);
  loop(state);
  return 0;
}
