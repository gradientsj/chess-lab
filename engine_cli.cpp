// Command-line bridge to the engine in chess.hpp, used by the RL arena to
// play the alpha-beta engine from Python without linking anything.
//   Build: g++ -std=c++20 -O2 engine_cli.cpp -o engine_cli.exe
//   Protocol: one request per line, "<depth> <FEN>", answered with the
//   engine's chosen move in coordinate notation (e2e4, e7e8q), or "none"
//   when the side to move has no legal moves.

#include "chess.hpp"
#include <iostream>
#include <sstream>
#include <string>

static std::string toUci(const Move& m) {
    auto sq = [](int s) {
        std::string out;
        out += char('a' + colOf(s));
        out += char('8' - rowOf(s));
        return out;
    };
    std::string out = sq(m.from) + sq(m.to);
    switch (m.promo) {
        case PieceType::Knight: out += 'n'; break;
        case PieceType::Bishop: out += 'b'; break;
        case PieceType::Rook:   out += 'r'; break;
        case PieceType::Queen:  out += 'q'; break;
        default: break;
    }
    return out;
}

int main() {
    std::string line;
    while (std::getline(std::cin, line)) {
        // Tolerate byte-order marks and stray carriage returns from callers
        // that pipe text through shells; a request always starts with the depth.
        while (!line.empty() && (line.front() < '0' || line.front() > '9')) line.erase(0, 1);
        while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) line.pop_back();
        if (line.empty()) continue;
        std::istringstream in(line);
        int depth = 1;
        in >> depth;
        std::string fen;
        std::getline(in, fen);
        if (!fen.empty() && fen.front() == ' ') fen.erase(0, 1);

        GameState s = parseFEN(fen);
        Move m = findBestMove(s, depth, 0);
        std::cout << (m.valid() ? toUci(m) : std::string("none")) << std::endl;
    }
    return 0;
}
