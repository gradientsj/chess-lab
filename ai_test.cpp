// Sanity checks for evaluation and search in chess.hpp.
//   Build: g++ -std=c++20 -O2 ai_test.cpp -o ai_test.exe
#include "chess.hpp"
#include <cstdio>
#include <sstream>
#include <string>

GameState parseFEN(const std::string& fen) {
    GameState s;
    std::istringstream in(fen);
    std::string placement, side, castling, ep;
    in >> placement >> side >> castling >> ep;
    int r = 0, c = 0;
    auto pt = [](char lc) {
        switch (lc) { case 'p': return PieceType::Pawn; case 'n': return PieceType::Knight;
                      case 'b': return PieceType::Bishop; case 'r': return PieceType::Rook;
                      case 'q': return PieceType::Queen; case 'k': return PieceType::King;
                      default: return PieceType::None; } };
    for (char ch : placement) {
        if (ch == '/') { ++r; c = 0; }
        else if (ch >= '1' && ch <= '8') c += ch - '0';
        else { Color col = (ch >= 'a') ? Color::Black : Color::White;
               char lc = (ch >= 'a') ? ch : char(ch - 'A' + 'a');
               s.board[sqOf(r, c)] = {pt(lc), col}; ++c; } }
    s.side = (side == "w") ? Color::White : Color::Black;
    s.castleWK = castling.find('K') != std::string::npos;
    s.castleWQ = castling.find('Q') != std::string::npos;
    s.castleBK = castling.find('k') != std::string::npos;
    s.castleBQ = castling.find('q') != std::string::npos;
    return s;
}

std::string sqName(int sq) { return std::string(1, char('a' + colOf(sq))) + char('0' + (8 - rowOf(sq))); }
std::string moveName(const Move& m) { return m.valid() ? sqName(m.from) + sqName(m.to) : "(none)"; }

int failures = 0;
void expect(const std::string& name, const std::string& got, const std::string& want) {
    bool ok = got == want;
    if (!ok) ++failures;
    std::printf("  %-22s got %-6s %s\n", name.c_str(), got.c_str(),
                ok ? "ok" : ("FAIL want " + want).c_str());
}

int main() {
    std::printf("AI / evaluation sanity checks\n\n");

    // Starting position is materially symmetric.
    expect("eval(start)==0", std::to_string(evaluate(initialPosition())), "0");

    // Back-rank mate in one: Ra1-a8#.
    {
        GameState s = parseFEN("6k1/5ppp/8/8/8/8/8/R6K w - -");
        expect("mate-in-1", moveName(findBestMove(s, 2, 0)), "a1a8");
    }

    // Free queen hanging: e4 pawn captures d5 queen.
    {
        GameState s = parseFEN("4k3/8/8/3q4/4P3/8/8/4K3 w - -");
        expect("win hanging queen", moveName(findBestMove(s, 3, 0)), "e4d5");
    }

    // A short AI-vs-AI game must terminate without crashing.
    {
        GameState s = initialPosition();
        std::vector<std::string> hist{positionKey(s)};
        int plies = 0; bool ended = false;
        for (; plies < 120; ++plies) {
            if (evaluateGameEnd(s, hist).over) { ended = true; break; }
            Move m = findBestMove(s, 2, 0);
            if (!m.valid()) { ended = true; break; }
            s = makeMove(s, m);
            hist.push_back(positionKey(s));
        }
        std::printf("  %-22s %d plies, %s\n", "self-play runs",
                    plies, ended ? "reached an end/limit (ok)" : "still going (ok)");
    }

    std::printf("\n%s\n", failures == 0 ? "ALL AI TESTS PASSED" : (std::to_string(failures) + " FAILED").c_str());
    return failures == 0 ? 0 : 1;
}
