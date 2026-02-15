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
      --rank; file = 0; continue;
    }
    if (std::isdigit(static_cast<unsigned char>(c))) { file += c - '0'; continue; }
    if (rank < 0 || file > 7) return false;
    squares[rank * 8 + file++] = c;
  }
  if (rank != 0 || file != 8) return false;
  whiteToMove = (side == "w");
  if (side != "w" && side != "b") return false;
  if (castling.find('K') != std::string::npos) castlingRights |= 1;
  if (castling.find('Q') != std::string::npos) castlingRights |= 2;
  if (castling.find('k') != std::string::npos) castlingRights |= 4;
  if (castling.find('q') != std::string::npos) castlingRights |= 8;
  enPassantSquare = (ep == "-") ? -1 : squareIndex(ep[0], ep[1]);
  return true;
}

bool Board::isSquareAttacked(int sq, bool byWhite) const {
  int f = sq % 8, r = sq / 8;
  int pawnDir = byWhite ? -1 : 1;
  for (int df : {-1, 1}) {
    int nf = f + df, nr = r + pawnDir;
    if (nf >= 0 && nf < 8 && nr >= 0 && nr < 8) {
      if (squares[nr * 8 + nf] == (byWhite ? 'P' : 'p')) return true;
    }
  }
  static const int kn[8][2] = {{1,2},{2,1},{2,-1},{1,-2},{-1,-2},{-2,-1},{-2,1},{-1,2}};
  for (auto& d : kn) {
    int nf = f + d[0], nr = r + d[1];
    if (nf >= 0 && nf < 8 && nr >= 0 && nr < 8) {
      if (squares[nr * 8 + nf] == (byWhite ? 'N' : 'n')) return true;
    }
  }
  auto ray = [&](int df, int dr, const char* set) {
    int nf = f + df, nr = r + dr;
    while (nf >= 0 && nf < 8 && nr >= 0 && nr < 8) {
      char p = squares[nr * 8 + nf];
      if (p != '.') {
        if (std::isupper(static_cast<unsigned char>(p)) == byWhite) {
          char q = static_cast<char>(std::tolower(static_cast<unsigned char>(p)));
          for (int i = 0; set[i]; ++i) if (set[i] == q) return true;
        }
        return false;
      }
      nf += df; nr += dr;
    }
    return false;
  };
  if (ray(1,0,"rq") || ray(-1,0,"rq") || ray(0,1,"rq") || ray(0,-1,"rq")) return true;
  if (ray(1,1,"bq") || ray(-1,1,"bq") || ray(1,-1,"bq") || ray(-1,-1,"bq")) return true;

  for (int df = -1; df <= 1; ++df) for (int dr = -1; dr <= 1; ++dr) if (df || dr) {
    int nf = f + df, nr = r + dr;
    if (nf >= 0 && nf < 8 && nr >= 0 && nr < 8 && squares[nr * 8 + nf] == (byWhite ? 'K' : 'k')) return true;
  }
  return false;
}

bool Board::inCheck(bool white) const {
  char king = white ? 'K' : 'k';
  for (int i = 0; i < 64; ++i) {
    if (squares[i] == king) return isSquareAttacked(i, !white);
  }
  return false;
}

bool Board::makeMove(int from, int to, char promotion, Undo& u) {
  u = Undo{};
  u.prevEnPassant = enPassantSquare;
  u.prevCastling = castlingRights;
  u.prevHalfmove = halfmoveClock;
  u.prevFullmove = fullmoveNumber;
  u.prevWhiteToMove = whiteToMove;
  u.moved = squares[from];
  u.captured = squares[to];
  u.capturedSquare = to;
  if (u.moved == '.') return false;

  bool movingWhite = std::isupper(static_cast<unsigned char>(u.moved));
  if (movingWhite != whiteToMove) return false;

  enPassantSquare = -1;
  if (std::tolower(static_cast<unsigned char>(u.moved)) == 'p' || u.captured != '.') halfmoveClock = 0;
  else ++halfmoveClock;

  if (std::tolower(static_cast<unsigned char>(u.moved)) == 'p' && to == u.prevEnPassant && u.captured == '.') {
    u.wasEnPassant = true;
    u.capturedSquare = to + (movingWhite ? -8 : 8);
    u.captured = squares[u.capturedSquare];
    squares[u.capturedSquare] = '.';
  }

  squares[to] = u.moved;
  squares[from] = '.';

  if (std::tolower(static_cast<unsigned char>(u.moved)) == 'p') {
    if (std::abs(to - from) == 16) enPassantSquare = (to + from) / 2;
    int toRank = to / 8;
    if ((movingWhite && toRank == 7) || (!movingWhite && toRank == 0)) {
      char promo = promotion ? promotion : 'q';
      squares[to] = movingWhite ? static_cast<char>(std::toupper(promo)) : static_cast<char>(std::tolower(promo));
      u.wasPromotion = true;
    }
  }

  if (u.moved == 'K') castlingRights &= ~(1u | 2u);
  if (u.moved == 'k') castlingRights &= ~(4u | 8u);
  if (from == 0 || to == 0) castlingRights &= ~2u;
  if (from == 7 || to == 7) castlingRights &= ~1u;
  if (from == 56 || to == 56) castlingRights &= ~8u;
  if (from == 63 || to == 63) castlingRights &= ~4u;

  if (std::tolower(static_cast<unsigned char>(u.moved)) == 'k' && std::abs(to - from) == 2) {
    u.wasCastle = true;
    if (to == 6) { squares[5] = 'R'; squares[7] = '.'; }
    else if (to == 2) { squares[3] = 'R'; squares[0] = '.'; }
    else if (to == 62) { squares[61] = 'r'; squares[63] = '.'; }
    else if (to == 58) { squares[59] = 'r'; squares[56] = '.'; }
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
  (void)promotion;
  whiteToMove = u.prevWhiteToMove;
  castlingRights = u.prevCastling;
  enPassantSquare = u.prevEnPassant;
  halfmoveClock = u.prevHalfmove;
  fullmoveNumber = u.prevFullmove;

  if (u.wasCastle) {
    if (to == 6) { squares[7] = 'R'; squares[5] = '.'; }
    else if (to == 2) { squares[0] = 'R'; squares[3] = '.'; }
    else if (to == 62) { squares[63] = 'r'; squares[61] = '.'; }
    else if (to == 58) { squares[56] = 'r'; squares[59] = '.'; }
  }

  squares[from] = u.moved;
  squares[to] = '.';
  if (u.capturedSquare >= 0) squares[u.capturedSquare] = u.captured;
}

}  // namespace board
