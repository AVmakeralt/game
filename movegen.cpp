#include "movegen.h"

#include <cctype>

namespace movegen {

std::string Move::toUCI() const {
  if (from < 0 || to < 0) return "0000";
  std::string out;
  out.push_back(static_cast<char>('a' + from % 8));
  out.push_back(static_cast<char>('1' + from / 8));
  out.push_back(static_cast<char>('a' + to % 8));
  out.push_back(static_cast<char>('1' + to / 8));
  if (promotion) out.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(promotion))));
  return out;
}

bool parseUCIMove(const std::string& text, Move& out) {
  if (text.size() < 4) return false;
  out.from = board::Board::squareIndex(text[0], text[1]);
  out.to = board::Board::squareIndex(text[2], text[3]);
  out.promotion = text.size() >= 5 ? text[4] : '\0';
  return out.from >= 0 && out.to >= 0;
}

static bool sameSide(char a, char b) {
  if (a == '.' || b == '.') return false;
  return std::isupper(static_cast<unsigned char>(a)) == std::isupper(static_cast<unsigned char>(b));
}

static bool isKing(char c) {
  return std::tolower(static_cast<unsigned char>(c)) == 'k';
}

static void pushPawnMove(std::vector<Move>& out, int from, int to, bool promotionRank) {
  if (!promotionRank) {
    out.push_back({from, to, '\0'});
  } else {
    out.push_back({from, to, 'q'});
    out.push_back({from, to, 'r'});
    out.push_back({from, to, 'b'});
    out.push_back({from, to, 'n'});
  }
}

std::vector<Move> generatePseudoLegal(const board::Board& b) {
  std::vector<Move> moves;
  for (int from = 0; from < 64; ++from) {
    char piece = b.squares[from];
    if (piece == '.') continue;
    bool white = std::isupper(static_cast<unsigned char>(piece));
    if (white != b.whiteToMove) continue;
    int r = from / 8, f = from % 8;
    char p = static_cast<char>(std::tolower(static_cast<unsigned char>(piece)));

    if (p == 'p') {
      int dir = white ? 1 : -1;
      int one = from + dir * 8;
      if (one >= 0 && one < 64 && b.squares[one] == '.') {
        pushPawnMove(moves, from, one, (white && one / 8 == 7) || (!white && one / 8 == 0));
        int two = from + dir * 16;
        if ((white ? r == 1 : r == 6) && b.squares[two] == '.') moves.push_back({from, two, '\0'});
      }
      for (int df : {-1, 1}) {
        int nf = f + df;
        if (nf < 0 || nf > 7) continue;
        int to = one + df;
        if (to < 0 || to >= 64) continue;
        if (((b.squares[to] != '.' && !sameSide(piece, b.squares[to]) && !isKing(b.squares[to])) || to == b.enPassantSquare)) {
          pushPawnMove(moves, from, to, (white && to / 8 == 7) || (!white && to / 8 == 0));
        }
      }
      continue;
    }

    if (p == 'n') {
      static const int d[8][2] = {{1,2},{2,1},{2,-1},{1,-2},{-1,-2},{-2,-1},{-2,1},{-1,2}};
      for (auto& k : d) {
        int nf = f + k[0], nr = r + k[1];
        if (nf < 0 || nf > 7 || nr < 0 || nr > 7) continue;
        int to = nr * 8 + nf;
        if (!sameSide(piece, b.squares[to]) && !isKing(b.squares[to])) moves.push_back({from, to, '\0'});
      }
      continue;
    }

    auto slide = [&](const int dirs[][2], int count, bool single) {
      for (int i = 0; i < count; ++i) {
        int nf = f + dirs[i][0], nr = r + dirs[i][1];
        while (nf >= 0 && nf < 8 && nr >= 0 && nr < 8) {
          int to = nr * 8 + nf;
          if (sameSide(piece, b.squares[to])) break;
          if (isKing(b.squares[to])) break;
          moves.push_back({from, to, '\0'});
          if (b.squares[to] != '.' || single) break;
          nf += dirs[i][0];
          nr += dirs[i][1];
        }
      }
    };

    static const int bdirs[4][2] = {{1,1},{1,-1},{-1,1},{-1,-1}};
    static const int rdirs[4][2] = {{1,0},{-1,0},{0,1},{0,-1}};
    static const int kdirs[8][2] = {{1,1},{1,0},{1,-1},{0,1},{0,-1},{-1,1},{-1,0},{-1,-1}};

    if (p == 'b') slide(bdirs, 4, false);
    else if (p == 'r') slide(rdirs, 4, false);
    else if (p == 'q') {
      slide(bdirs, 4, false);
      slide(rdirs, 4, false);
    } else if (p == 'k') {
      slide(kdirs, 8, true);
      if (white && from == 4 && piece == 'K') {
        if ((b.castlingRights & 1) && b.squares[7] == 'R' && b.squares[5] == '.' && b.squares[6] == '.' &&
            !b.isSquareAttacked(4, false) && !b.isSquareAttacked(5, false) && !b.isSquareAttacked(6, false))
          moves.push_back({4, 6, '\0'});
        if ((b.castlingRights & 2) && b.squares[0] == 'R' && b.squares[3] == '.' && b.squares[2] == '.' && b.squares[1] == '.' &&
            !b.isSquareAttacked(4, false) && !b.isSquareAttacked(3, false) && !b.isSquareAttacked(2, false))
          moves.push_back({4, 2, '\0'});
      } else if (!white && from == 60 && piece == 'k') {
        if ((b.castlingRights & 4) && b.squares[63] == 'r' && b.squares[61] == '.' && b.squares[62] == '.' &&
            !b.isSquareAttacked(60, true) && !b.isSquareAttacked(61, true) && !b.isSquareAttacked(62, true))
          moves.push_back({60, 62, '\0'});
        if ((b.castlingRights & 8) && b.squares[56] == 'r' && b.squares[59] == '.' && b.squares[58] == '.' && b.squares[57] == '.' &&
            !b.isSquareAttacked(60, true) && !b.isSquareAttacked(59, true) && !b.isSquareAttacked(58, true))
          moves.push_back({60, 58, '\0'});
      }
    }
  }
  return moves;
}

std::vector<Move> generateLegal(const board::Board& b) {
  auto pseudo = generatePseudoLegal(b);
  std::vector<Move> legal;
  board::Board copy = b;
  for (const auto& m : pseudo) {
    board::Undo u;
    if (copy.makeMove(m.from, m.to, m.promotion, u)) {
      legal.push_back(m);
      copy.unmakeMove(m.from, m.to, m.promotion, u);
    }
  }
  return legal;
}

bool isLegalMove(const board::Board& b, const Move& m) {
  auto legal = generateLegal(b);
  for (const auto& x : legal)
    if (x == m) return true;
  return false;
}

}  // namespace movegen
