// Chess engine: board, rules, move generation, evaluation, and search.
// Header-only and UI-independent so it can be reused and unit-tested (see perft.cpp).
#pragma once

#include <array>
#include <vector>
#include <string>
#include <random>
#include <algorithm>
#include <cstdint>

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------
enum class PieceType : std::uint8_t { None = 0, Pawn, Knight, Bishop, Rook, Queen, King };
enum class Color     : std::uint8_t { None = 0, White, Black };

inline Color opposite(Color c) { return c == Color::White ? Color::Black : Color::White; }

struct Piece {
    PieceType type  = PieceType::None;
    Color     color = Color::None;
    bool empty() const { return type == PieceType::None; }
};

enum class MoveFlag : std::uint8_t { Normal, DoublePush, EnPassant, Castle };

struct Move {
    int       from  = -1;
    int       to    = -1;
    PieceType promo = PieceType::None;   // promotion target, or None
    MoveFlag  flag  = MoveFlag::Normal;
    bool valid() const { return from >= 0; }
    bool operator==(const Move& m) const {
        return from == m.from && to == m.to && promo == m.promo;
    }
};

struct GameState {
    std::array<Piece, 64> board{};
    Color side = Color::White;          // side to move
    bool  castleWK = false, castleWQ = false, castleBK = false, castleBQ = false;
    int   epSquare = -1;                // en-passant target square, or -1
    int   halfmove = 0;                 // halfmove clock for the 50-move rule

    Piece at(int sq) const { return board[sq]; }
};

// ---------------------------------------------------------------------------
// Small board helpers (sq = row*8 + col, row 0 = rank 8 / top, col 0 = file a)
// ---------------------------------------------------------------------------
inline bool inside(int r, int c) { return r >= 0 && r < 8 && c >= 0 && c < 8; }
inline int  rowOf(int sq) { return sq / 8; }
inline int  colOf(int sq) { return sq % 8; }
inline int  sqOf(int r, int c) { return r * 8 + c; }

inline GameState initialPosition() {
    GameState s;
    auto place = [&](int r, int c, PieceType t, Color col) { s.board[sqOf(r, c)] = {t, col}; };
    const PieceType back[8] = {PieceType::Rook, PieceType::Knight, PieceType::Bishop, PieceType::Queen,
                               PieceType::King, PieceType::Bishop, PieceType::Knight, PieceType::Rook};
    for (int c = 0; c < 8; ++c) {
        place(0, c, back[c], Color::Black);
        place(1, c, PieceType::Pawn, Color::Black);
        place(6, c, PieceType::Pawn, Color::White);
        place(7, c, back[c], Color::White);
    }
    s.side = Color::White;
    s.castleWK = s.castleWQ = s.castleBK = s.castleBQ = true;
    s.epSquare = -1;
    s.halfmove = 0;
    return s;
}

// ---------------------------------------------------------------------------
// Attack detection
// ---------------------------------------------------------------------------
static const int knightOff[8][2] = {{-2,-1},{-2,1},{-1,-2},{-1,2},{1,-2},{1,2},{2,-1},{2,1}};
static const int kingOff[8][2]   = {{-1,-1},{-1,0},{-1,1},{0,-1},{0,1},{1,-1},{1,0},{1,1}};
static const int diagDir[4][2]   = {{-1,-1},{-1,1},{1,-1},{1,1}};
static const int orthoDir[4][2]  = {{-1,0},{1,0},{0,-1},{0,1}};

// Is square `sq` attacked by any piece of color `by`?
inline bool isSquareAttacked(const GameState& s, int sq, Color by) {
    int r = rowOf(sq), c = colOf(sq);

    // Pawns: a `by`-pawn that attacks sq sits one row toward its own side.
    int pawnRow = (by == Color::White) ? r + 1 : r - 1;  // white pawns sit below sq
    for (int dc : {-1, 1}) {
        int pr = pawnRow, pc = c + dc;
        if (inside(pr, pc)) {
            Piece p = s.board[sqOf(pr, pc)];
            if (p.type == PieceType::Pawn && p.color == by) return true;
        }
    }
    for (auto& o : knightOff) {
        int nr = r + o[0], nc = c + o[1];
        if (inside(nr, nc)) {
            Piece p = s.board[sqOf(nr, nc)];
            if (p.type == PieceType::Knight && p.color == by) return true;
        }
    }
    for (auto& o : kingOff) {
        int nr = r + o[0], nc = c + o[1];
        if (inside(nr, nc)) {
            Piece p = s.board[sqOf(nr, nc)];
            if (p.type == PieceType::King && p.color == by) return true;
        }
    }
    for (auto& d : diagDir) {
        int nr = r + d[0], nc = c + d[1];
        while (inside(nr, nc)) {
            Piece p = s.board[sqOf(nr, nc)];
            if (!p.empty()) {
                if (p.color == by && (p.type == PieceType::Bishop || p.type == PieceType::Queen))
                    return true;
                break;
            }
            nr += d[0]; nc += d[1];
        }
    }
    for (auto& d : orthoDir) {
        int nr = r + d[0], nc = c + d[1];
        while (inside(nr, nc)) {
            Piece p = s.board[sqOf(nr, nc)];
            if (!p.empty()) {
                if (p.color == by && (p.type == PieceType::Rook || p.type == PieceType::Queen))
                    return true;
                break;
            }
            nr += d[0]; nc += d[1];
        }
    }
    return false;
}

inline int findKing(const GameState& s, Color c) {
    for (int sq = 0; sq < 64; ++sq)
        if (s.board[sq].type == PieceType::King && s.board[sq].color == c) return sq;
    return -1;
}

inline bool inCheck(const GameState& s, Color c) {
    int k = findKing(s, c);
    return k >= 0 && isSquareAttacked(s, k, opposite(c));
}

// ---------------------------------------------------------------------------
// Move generation
// ---------------------------------------------------------------------------
inline void addPawnMoves(std::vector<Move>& out, int from, int to, MoveFlag flag, int promoRow) {
    if (rowOf(to) == promoRow) {
        for (PieceType pt : {PieceType::Queen, PieceType::Rook, PieceType::Bishop, PieceType::Knight})
            out.push_back({from, to, pt, flag});
    } else {
        out.push_back({from, to, PieceType::None, flag});
    }
}

inline void genPseudoLegal(const GameState& s, std::vector<Move>& out) {
    Color us = s.side, them = opposite(us);
    int dir      = (us == Color::White) ? -1 : +1;   // forward = toward smaller row for White
    int startRow = (us == Color::White) ? 6 : 1;
    int promoRow = (us == Color::White) ? 0 : 7;

    for (int sq = 0; sq < 64; ++sq) {
        Piece p = s.board[sq];
        if (p.empty() || p.color != us) continue;
        int r = rowOf(sq), c = colOf(sq);

        switch (p.type) {
        case PieceType::Pawn: {
            int nr = r + dir;
            if (inside(nr, c) && s.board[sqOf(nr, c)].empty()) {
                addPawnMoves(out, sq, sqOf(nr, c), MoveFlag::Normal, promoRow);
                int nr2 = r + 2 * dir;
                if (r == startRow && s.board[sqOf(nr2, c)].empty())
                    out.push_back({sq, sqOf(nr2, c), PieceType::None, MoveFlag::DoublePush});
            }
            for (int dc : {-1, 1}) {
                int cc = c + dc;
                if (!inside(nr, cc)) continue;
                int t = sqOf(nr, cc);
                Piece tp = s.board[t];
                if (!tp.empty() && tp.color == them)
                    addPawnMoves(out, sq, t, MoveFlag::Normal, promoRow);
                else if (t == s.epSquare)
                    out.push_back({sq, t, PieceType::None, MoveFlag::EnPassant});
            }
            break;
        }
        case PieceType::Knight:
            for (auto& o : knightOff) {
                int nr = r + o[0], nc = c + o[1];
                if (!inside(nr, nc)) continue;
                Piece tp = s.board[sqOf(nr, nc)];
                if (tp.empty() || tp.color == them) out.push_back({sq, sqOf(nr, nc)});
            }
            break;
        case PieceType::King:
            for (auto& o : kingOff) {
                int nr = r + o[0], nc = c + o[1];
                if (!inside(nr, nc)) continue;
                Piece tp = s.board[sqOf(nr, nc)];
                if (tp.empty() || tp.color == them) out.push_back({sq, sqOf(nr, nc)});
            }
            {
                int kr = (us == Color::White) ? 7 : 0;
                bool canK = (us == Color::White) ? s.castleWK : s.castleBK;
                bool canQ = (us == Color::White) ? s.castleWQ : s.castleBQ;
                if (sq == sqOf(kr, 4) && !inCheck(s, us)) {
                    if (canK && s.board[sqOf(kr, 5)].empty() && s.board[sqOf(kr, 6)].empty()
                        && !isSquareAttacked(s, sqOf(kr, 5), them)
                        && !isSquareAttacked(s, sqOf(kr, 6), them))
                        out.push_back({sq, sqOf(kr, 6), PieceType::None, MoveFlag::Castle});
                    if (canQ && s.board[sqOf(kr, 3)].empty() && s.board[sqOf(kr, 2)].empty()
                        && s.board[sqOf(kr, 1)].empty()
                        && !isSquareAttacked(s, sqOf(kr, 3), them)
                        && !isSquareAttacked(s, sqOf(kr, 2), them))
                        out.push_back({sq, sqOf(kr, 2), PieceType::None, MoveFlag::Castle});
                }
            }
            break;
        case PieceType::Bishop:
        case PieceType::Rook:
        case PieceType::Queen: {
            const int (*dirs)[2]; int ndir;
            if (p.type == PieceType::Bishop)      { dirs = diagDir;  ndir = 4; }
            else if (p.type == PieceType::Rook)   { dirs = orthoDir; ndir = 4; }
            else {  // queen: walk both diagonal and orthogonal rays
                for (int set = 0; set < 2; ++set) {
                    const int (*dd)[2] = set ? orthoDir : diagDir;
                    for (int i = 0; i < 4; ++i) {
                        int nr = r + dd[i][0], nc = c + dd[i][1];
                        while (inside(nr, nc)) {
                            Piece tp = s.board[sqOf(nr, nc)];
                            if (tp.empty()) out.push_back({sq, sqOf(nr, nc)});
                            else { if (tp.color == them) out.push_back({sq, sqOf(nr, nc)}); break; }
                            nr += dd[i][0]; nc += dd[i][1];
                        }
                    }
                }
                break;
            }
            for (int i = 0; i < ndir; ++i) {
                int nr = r + dirs[i][0], nc = c + dirs[i][1];
                while (inside(nr, nc)) {
                    Piece tp = s.board[sqOf(nr, nc)];
                    if (tp.empty()) out.push_back({sq, sqOf(nr, nc)});
                    else { if (tp.color == them) out.push_back({sq, sqOf(nr, nc)}); break; }
                    nr += dirs[i][0]; nc += dirs[i][1];
                }
            }
            break;
        }
        default: break;
        }
    }
}

// Apply a move and return the resulting state (copy-based — board is tiny).
inline GameState makeMove(const GameState& s, const Move& m) {
    GameState n = s;
    Color us = s.side, them = opposite(us);
    Piece moving = n.board[m.from];
    bool capture = !n.board[m.to].empty();

    n.epSquare = -1;

    if (m.flag == MoveFlag::EnPassant) {
        int capSq = (us == Color::White) ? m.to + 8 : m.to - 8;  // captured pawn sits one row back
        n.board[capSq] = {};
        capture = true;
    }

    n.board[m.to] = moving;
    n.board[m.from] = {};

    if (m.promo != PieceType::None) n.board[m.to].type = m.promo;

    if (m.flag == MoveFlag::DoublePush) n.epSquare = (m.from + m.to) / 2;

    if (m.flag == MoveFlag::Castle) {
        int kr = rowOf(m.to);
        if (colOf(m.to) == 6) { n.board[sqOf(kr, 5)] = n.board[sqOf(kr, 7)]; n.board[sqOf(kr, 7)] = {}; }
        else                  { n.board[sqOf(kr, 3)] = n.board[sqOf(kr, 0)]; n.board[sqOf(kr, 0)] = {}; }
    }

    if (moving.type == PieceType::King) {
        if (us == Color::White) { n.castleWK = n.castleWQ = false; }
        else                    { n.castleBK = n.castleBQ = false; }
    }
    auto touch = [&](int sq) {
        if (sq == sqOf(7, 0)) n.castleWQ = false;
        if (sq == sqOf(7, 7)) n.castleWK = false;
        if (sq == sqOf(0, 0)) n.castleBQ = false;
        if (sq == sqOf(0, 7)) n.castleBK = false;
    };
    touch(m.from);  // a rook left its corner
    touch(m.to);    // a corner rook was captured

    n.halfmove = (moving.type == PieceType::Pawn || capture) ? 0 : s.halfmove + 1;
    n.side = them;
    return n;
}

// Legal moves = pseudo-legal moves that don't leave our own king in check.
inline std::vector<Move> generateLegal(const GameState& s) {
    std::vector<Move> pseudo, legal;
    pseudo.reserve(64);
    genPseudoLegal(s, pseudo);
    Color us = s.side;
    for (const Move& m : pseudo) {
        GameState n = makeMove(s, m);
        if (!inCheck(n, us)) legal.push_back(m);
    }
    return legal;
}

// ---------------------------------------------------------------------------
// Game-end detection
// ---------------------------------------------------------------------------
inline std::string positionKey(const GameState& s) {
    std::string k;
    k.reserve(72);
    const char* w = " PNBRQK";
    const char* b = " pnbrqk";
    for (int sq = 0; sq < 64; ++sq) {
        Piece p = s.board[sq];
        if (p.empty()) k += '.';
        else k += (p.color == Color::White ? w : b)[(int)p.type];
    }
    k += (s.side == Color::White ? 'w' : 'b');
    k += s.castleWK ? 'K' : '-';
    k += s.castleWQ ? 'Q' : '-';
    k += s.castleBK ? 'k' : '-';
    k += s.castleBQ ? 'q' : '-';
    k += (s.epSquare < 0 ? '-' : char('a' + colOf(s.epSquare)));
    return k;
}

inline bool insufficientMaterial(const GameState& s) {
    int minors = 0;
    for (int sq = 0; sq < 64; ++sq) {
        Piece p = s.board[sq];
        switch (p.type) {
        case PieceType::Pawn: case PieceType::Rook: case PieceType::Queen: return false;
        case PieceType::Bishop: case PieceType::Knight: ++minors; break;
        default: break;
        }
    }
    return minors <= 1;  // K vs K, K+minor vs K
}

struct GameEnd { bool over = false; std::string message; };

inline GameEnd evaluateGameEnd(const GameState& s, const std::vector<std::string>& history) {
    if (generateLegal(s).empty()) {
        if (inCheck(s, s.side)) {
            std::string winner = (s.side == Color::White) ? "Black" : "White";
            return {true, "Checkmate — " + winner + " wins!"};
        }
        return {true, "Stalemate — draw"};
    }
    if (s.halfmove >= 100) return {true, "Draw — 50-move rule"};
    if (insufficientMaterial(s)) return {true, "Draw — insufficient material"};
    if (!history.empty()) {
        const std::string& cur = history.back();
        int count = 0;
        for (const auto& k : history) if (k == cur) ++count;
        if (count >= 3) return {true, "Draw — threefold repetition"};
    }
    return {false, ""};
}

// ---------------------------------------------------------------------------
// Evaluation + search (negamax / alpha-beta / quiescence)
// ---------------------------------------------------------------------------
static const int MATE = 1000000;
static const int INF  = 1000000000;
static const int matValue[7] = {0, 100, 320, 330, 500, 900, 0};

// Piece-square tables, index 0 = a8 (top-left), oriented for White.
static const int PST[6][64] = {
    {  0,  0,  0,  0,  0,  0,  0,  0,
      50, 50, 50, 50, 50, 50, 50, 50,
      10, 10, 20, 30, 30, 20, 10, 10,
       5,  5, 10, 25, 25, 10,  5,  5,
       0,  0,  0, 20, 20,  0,  0,  0,
       5, -5,-10,  0,  0,-10, -5,  5,
       5, 10, 10,-20,-20, 10, 10,  5,
       0,  0,  0,  0,  0,  0,  0,  0 },
    {-50,-40,-30,-30,-30,-30,-40,-50,
     -40,-20,  0,  0,  0,  0,-20,-40,
     -30,  0, 10, 15, 15, 10,  0,-30,
     -30,  5, 15, 20, 20, 15,  5,-30,
     -30,  0, 15, 20, 20, 15,  0,-30,
     -30,  5, 10, 15, 15, 10,  5,-30,
     -40,-20,  0,  5,  5,  0,-20,-40,
     -50,-40,-30,-30,-30,-30,-40,-50 },
    {-20,-10,-10,-10,-10,-10,-10,-20,
     -10,  0,  0,  0,  0,  0,  0,-10,
     -10,  0,  5, 10, 10,  5,  0,-10,
     -10,  5,  5, 10, 10,  5,  5,-10,
     -10,  0, 10, 10, 10, 10,  0,-10,
     -10, 10, 10, 10, 10, 10, 10,-10,
     -10,  5,  0,  0,  0,  0,  5,-10,
     -20,-10,-10,-10,-10,-10,-10,-20 },
    {  0,  0,  0,  0,  0,  0,  0,  0,
       5, 10, 10, 10, 10, 10, 10,  5,
      -5,  0,  0,  0,  0,  0,  0, -5,
      -5,  0,  0,  0,  0,  0,  0, -5,
      -5,  0,  0,  0,  0,  0,  0, -5,
      -5,  0,  0,  0,  0,  0,  0, -5,
      -5,  0,  0,  0,  0,  0,  0, -5,
       0,  0,  0,  5,  5,  0,  0,  0 },
    {-20,-10,-10, -5, -5,-10,-10,-20,
     -10,  0,  0,  0,  0,  0,  0,-10,
     -10,  0,  5,  5,  5,  5,  0,-10,
      -5,  0,  5,  5,  5,  5,  0, -5,
       0,  0,  5,  5,  5,  5,  0, -5,
     -10,  5,  5,  5,  5,  5,  0,-10,
     -10,  0,  5,  0,  0,  0,  0,-10,
     -20,-10,-10, -5, -5,-10,-10,-20 },
    {-30,-40,-40,-50,-50,-40,-40,-30,
     -30,-40,-40,-50,-50,-40,-40,-30,
     -30,-40,-40,-50,-50,-40,-40,-30,
     -30,-40,-40,-50,-50,-40,-40,-30,
     -20,-30,-30,-40,-40,-30,-30,-20,
     -10,-20,-20,-20,-20,-20,-20,-10,
      20, 20,  0,  0,  0,  0, 20, 20,
      20, 30, 10,  0,  0, 10, 30, 20 }
};

// Static evaluation, from White's perspective.
inline int evaluate(const GameState& s) {
    int score = 0;
    for (int sq = 0; sq < 64; ++sq) {
        Piece p = s.board[sq];
        if (p.empty()) continue;
        int ti = (int)p.type - 1;
        int pstSq = (p.color == Color::White) ? sq : (sq ^ 56);  // mirror rows for Black
        int v = matValue[(int)p.type] + PST[ti][pstSq];
        score += (p.color == Color::White) ? v : -v;
    }
    return score;
}

inline bool isCapture(const GameState& s, const Move& m) {
    return !s.board[m.to].empty() || m.flag == MoveFlag::EnPassant;
}

// Most-valuable-victim ordering: captures first, biggest victims first.
inline void orderMoves(const GameState& s, std::vector<Move>& moves) {
    std::stable_sort(moves.begin(), moves.end(), [&](const Move& a, const Move& b) {
        int va = isCapture(s, a) ? matValue[(int)s.board[a.to].type] + 1000 : 0;
        int vb = isCapture(s, b) ? matValue[(int)s.board[b.to].type] + 1000 : 0;
        if (a.promo != PieceType::None) va += matValue[(int)a.promo];
        if (b.promo != PieceType::None) vb += matValue[(int)b.promo];
        return va > vb;
    });
}

inline int quiescence(const GameState& s, int alpha, int beta, int qdepth) {
    int standPat = (s.side == Color::White) ? evaluate(s) : -evaluate(s);
    if (standPat >= beta) return beta;
    if (standPat > alpha) alpha = standPat;
    if (qdepth <= 0) return alpha;

    std::vector<Move> moves;
    genPseudoLegal(s, moves);
    std::vector<Move> caps;
    for (const Move& m : moves) if (isCapture(s, m)) caps.push_back(m);
    orderMoves(s, caps);

    for (const Move& m : caps) {
        GameState n = makeMove(s, m);
        if (inCheck(n, s.side)) continue;          // illegal — would leave king in check
        int val = -quiescence(n, -beta, -alpha, qdepth - 1);
        if (val >= beta) return beta;
        if (val > alpha) alpha = val;
    }
    return alpha;
}

inline int negamax(const GameState& s, int depth, int alpha, int beta, int ply) {
    std::vector<Move> moves = generateLegal(s);
    if (moves.empty())
        return inCheck(s, s.side) ? -(MATE - ply) : 0;   // checkmate : stalemate
    if (depth == 0)
        return quiescence(s, alpha, beta, 4);

    orderMoves(s, moves);
    int best = -INF;
    for (const Move& m : moves) {
        GameState n = makeMove(s, m);
        int val = -negamax(n, depth - 1, -beta, -alpha, ply + 1);
        if (val > best) best = val;
        if (best > alpha) alpha = best;
        if (alpha >= beta) break;
    }
    return best;
}

// Pick the AI's move. `randomMargin` widens the pool of "good enough" moves
// to pick from, making weaker levels less predictable.
inline Move findBestMove(GameState s, int depth, int randomMargin) {
    std::vector<Move> moves = generateLegal(s);
    if (moves.empty()) return {};
    orderMoves(s, moves);

    // Search each root move with a full window so every move gets its true
    // score. (A shared narrowing window would make pruned moves report the
    // alpha bound instead of their real value, corrupting the pick below.)
    // Deeper plies still prune with alpha-beta, which is where the savings are.
    int bestVal = -INF;
    std::vector<std::pair<int, Move>> scored;
    for (const Move& m : moves) {
        GameState n = makeMove(s, m);
        int val = -negamax(n, depth - 1, -INF, INF, 1);
        scored.push_back({val, m});
        if (val > bestVal) bestVal = val;
    }

    std::vector<Move> candidates;
    for (auto& [val, m] : scored)
        if (val >= bestVal - randomMargin) candidates.push_back(m);

    static std::mt19937 rng{std::random_device{}()};
    std::uniform_int_distribution<std::size_t> dist(0, candidates.size() - 1);
    return candidates[dist(rng)];
}
