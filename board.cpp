#include "board.h"

#include <cctype>
#include <sstream>

namespace board {

void Board::clear() {
  squares.fill('.');
  whiteToMove = true;
  castlingRights = 0;
  enPassantSquare = -1;
  halfmoveClock = 0;
  fullmoveNumber = 1;
  history.clear();
}

void Board::setStartPos() {
  setFromFEN("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
}

int Board::squareIndex(char file, char rank) {
  if (file < 'a' || file > 'h' || rank < '1' || rank > '8') return -1;
  return (rank - '1') * 8 + (file - 'a');
}

std::string Board::squareName(int sq) {
  if (sq < 0 || sq >= 64) return "-";
  return std::string{static_cast<char>('a' + sq % 8), static_cast<char>('1' + sq / 8)};
}

bool Board::setFromFEN(const std::string& fen) {
  clear();
  std::istringstream iss(fen);
  std::string placement, side, castling, ep;
  if (!(iss >> placement >> side >> castling >> ep >> halfmoveClock >> fullmoveNumber)) return false;

  int rank = 7, file = 0;
  for (char c : placement) {
    if (c == '/') {
      if (file != 8) return false;
      --rank;
      file = 0;
      continue;
    }
    if (std::isdigit(static_cast<unsigned char>(c))) {
      file += c - '0';
      continue;
    }
    if (rank < 0 || file > 7) return false;
    squares[rank * 8 + file++] = c;
  }
  if (rank != 0 || file != 8) return false;
  if (side != "w" && side != "b") return false;
  whiteToMove = side == "w";
  if (castling.find('K') != std::string::npos) castlingRights |= 1;
  if (castling.find('Q') != std::string::npos) castlingRights |= 2;
  if (castling.find('k') != std::string::npos) castlingRights |= 4;
  if (castling.find('q') != std::string::npos) castlingRights |= 8;
  enPassantSquare = (ep == "-") ? -1 : squareIndex(ep[0], ep[1]);
  return true;
}

bool Board::isSquareAttacked(int sq, bool byWhite) const {
  int f = sq % 8;
  int r = sq / 8;

  int pawnR = byWhite ? r - 1 : r + 1;
  if (pawnR >= 0 && pawnR < 8) {
    if (f - 1 >= 0 && squares[pawnR * 8 + (f - 1)] == (byWhite ? 'P' : 'p')) return true;
    if (f + 1 < 8 && squares[pawnR * 8 + (f + 1)] == (byWhite ? 'P' : 'p')) return true;
  }

  static const int kn[8][2] = {{1,2},{2,1},{2,-1},{1,-2},{-1,-2},{-2,-1},{-2,1},{-1,2}};
  for (auto& d : kn) {
    int nf = f + d[0], nr = r + d[1];
    if (nf >= 0 && nf < 8 && nr >= 0 && nr < 8 && squares[nr * 8 + nf] == (byWhite ? 'N' : 'n')) return true;
  }

  auto ray = [&](int df, int dr, const char* targets) {
    int nf = f + df, nr = r + dr;
    while (nf >= 0 && nf < 8 && nr >= 0 && nr < 8) {
      char p = squares[nr * 8 + nf];
      if (p != '.') {
        if (std::isupper(static_cast<unsigned char>(p)) == byWhite) {
          char q = static_cast<char>(std::tolower(static_cast<unsigned char>(p)));
          for (int i = 0; targets[i]; ++i) if (q == targets[i]) return true;
        }
        return false;
      }
      nf += df;
      nr += dr;
    }
    return false;
  };

  if (ray(1, 0, "rq") || ray(-1, 0, "rq") || ray(0, 1, "rq") || ray(0, -1, "rq")) return true;
  if (ray(1, 1, "bq") || ray(1, -1, "bq") || ray(-1, 1, "bq") || ray(-1, -1, "bq")) return true;

  for (int df = -1; df <= 1; ++df) {
    for (int dr = -1; dr <= 1; ++dr) {
      if (!df && !dr) continue;
      int nf = f + df, nr = r + dr;
      if (nf >= 0 && nf < 8 && nr >= 0 && nr < 8 && squares[nr * 8 + nf] == (byWhite ? 'K' : 'k')) return true;
    }
  }
  return false;
}

bool Board::inCheck(bool white) const {
  char k = white ? 'K' : 'k';
  for (int i = 0; i < 64; ++i) {
    if (squares[i] == k) return isSquareAttacked(i, !white);
  }
  return false;
}

bool Board::makeMove(int from, int to, char promotion, Undo& u) {
  if (from < 0 || from >= 64 || to < 0 || to >= 64) return false;
  char piece = squares[from];
  if (piece == '.') return false;

  bool movingWhite = std::isupper(static_cast<unsigned char>(piece));
  if (movingWhite != whiteToMove) return false;

  u.prevSquares = squares;
  u.prevWhiteToMove = whiteToMove;
  u.prevCastling = castlingRights;
  u.prevEnPassant = enPassantSquare;
  u.prevHalfmove = halfmoveClock;
  u.prevFullmove = fullmoveNumber;

  char captured = squares[to];
  if (std::tolower(static_cast<unsigned char>(captured)) == "k"[0]) return false;
  enPassantSquare = -1;
  if (std::tolower(static_cast<unsigned char>(piece)) == 'p' || captured != '.') halfmoveClock = 0;
  else ++halfmoveClock;

  if (std::tolower(static_cast<unsigned char>(piece)) == 'p' && to == u.prevEnPassant && captured == '.') {
    int capSq = to + (movingWhite ? -8 : 8);
    squares[capSq] = '.';
  }

  squares[to] = piece;
  squares[from] = '.';

  if (std::tolower(static_cast<unsigned char>(piece)) == 'p') {
    if (std::abs(to - from) == 16) enPassantSquare = (to + from) / 2;
    int rank = to / 8;
    if ((movingWhite && rank == 7) || (!movingWhite && rank == 0)) {
      char pp = promotion ? promotion : 'q';
      squares[to] = movingWhite ? static_cast<char>(std::toupper(pp)) : static_cast<char>(std::tolower(pp));
    }
  }

  if (piece == 'K') castlingRights &= ~(1u | 2u);
  if (piece == 'k') castlingRights &= ~(4u | 8u);
  if (from == 0 || to == 0) castlingRights &= ~2u;
  if (from == 7 || to == 7) castlingRights &= ~1u;
  if (from == 56 || to == 56) castlingRights &= ~8u;
  if (from == 63 || to == 63) castlingRights &= ~4u;

  if (std::tolower(static_cast<unsigned char>(piece)) == 'k' && std::abs(to - from) == 2) {
    if (to == 6) {
      squares[5] = 'R';
      squares[7] = '.';
    } else if (to == 2) {
      squares[3] = 'R';
      squares[0] = '.';
    } else if (to == 62) {
      squares[61] = 'r';
      squares[63] = '.';
    } else if (to == 58) {
      squares[59] = 'r';
      squares[56] = '.';
    }
  }

  whiteToMove = !whiteToMove;
  if (whiteToMove) ++fullmoveNumber;

  if (inCheck(!whiteToMove)) {
    unmakeMove(from, to, promotion, u);
    return false;
  }
  return true;
}

void Board::unmakeMove(int from, int to, char promotion, const Undo& u) {
  (void)from;
  (void)to;
  (void)promotion;
  squares = u.prevSquares;
  whiteToMove = u.prevWhiteToMove;
  castlingRights = u.prevCastling;
  enPassantSquare = u.prevEnPassant;
  halfmoveClock = u.prevHalfmove;
  fullmoveNumber = u.prevFullmove;
}

}  // namespace board
