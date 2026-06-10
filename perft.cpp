// Perft test for the chess engine in chess.hpp.
// Compares legal-move counts against published reference values; exact matches
// verify castling, en passant, promotion, and check handling.
//   Build: g++ -std=c++20 -O2 perft.cpp -o perft.exe

#include "chess.hpp"
#include <cstdio>
#include <cstdint>

std::uint64_t perft(const GameState& s, int depth) {
    if (depth == 0) return 1;
    std::uint64_t nodes = 0;
    for (const Move& m : generateLegal(s))
        nodes += perft(makeMove(s, m), depth - 1);
    return nodes;
}

int failures = 0;

void check(const std::string& name, const std::string& fen, int depth, std::uint64_t expected) {
    GameState s = parseFEN(fen);
    std::uint64_t got = perft(s, depth);
    bool ok = got == expected;
    if (!ok) ++failures;
    std::printf("  %-12s depth %d : %-10llu  %s\n",
                name.c_str(), depth, (unsigned long long)got,
                ok ? "ok" : ("FAIL expected " + std::to_string(expected)).c_str());
}

int main() {
    const std::string START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -";
    const std::string KIWI  = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -";
    const std::string POS3  = "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - -";
    const std::string POS4  = "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq -";
    const std::string POS5  = "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ -";

    std::printf("Perft verification (engine move generation)\n\n");

    check("start", START, 1, 20);
    check("start", START, 2, 400);
    check("start", START, 3, 8902);
    check("start", START, 4, 197281);
    check("start", START, 5, 4865609);

    check("kiwipete", KIWI, 1, 48);
    check("kiwipete", KIWI, 2, 2039);
    check("kiwipete", KIWI, 3, 97862);
    check("kiwipete", KIWI, 4, 4085603);

    check("position3", POS3, 1, 14);
    check("position3", POS3, 2, 191);
    check("position3", POS3, 3, 2812);
    check("position3", POS3, 4, 43238);
    check("position3", POS3, 5, 674624);

    check("position4", POS4, 1, 6);
    check("position4", POS4, 2, 264);
    check("position4", POS4, 3, 9467);
    check("position4", POS4, 4, 422333);

    check("position5", POS5, 1, 44);
    check("position5", POS5, 2, 1486);
    check("position5", POS5, 3, 62379);

    std::printf("\n%s\n", failures == 0 ? "ALL PERFT TESTS PASSED" : (std::to_string(failures) + " TEST(S) FAILED").c_str());
    return failures == 0 ? 0 : 1;
}
