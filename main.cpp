// C++ Chess — SFML 3.0 front-end.
// Human-vs-human and human-vs-AI, mouse drag-and-drop. Chess rules and the AI
// live in chess.hpp.  Build: build.ps1   Run: run.ps1

#include "chess.hpp"
#include <SFML/Graphics.hpp>
#include <optional>
#include <future>
#include <chrono>
#include <stdexcept>
#include <cstdio>

namespace layout {
    constexpr int SQUARE  = 80;
    constexpr int BOARD   = SQUARE * 8;       // 640
    constexpr int PANEL   = 220;
    constexpr int WIN_W   = BOARD + PANEL;    // 860
    constexpr int WIN_H   = BOARD;            // 640
}

enum class Screen { Menu, Playing };
enum class Mode   { HumanVsHuman, HumanVsAI };

inline char32_t glyphFor(PieceType t) {
    switch (t) {
        case PieceType::King:   return U'♚';
        case PieceType::Queen:  return U'♛';
        case PieceType::Rook:   return U'♜';
        case PieceType::Bishop: return U'♝';
        case PieceType::Knight: return U'♞';
        case PieceType::Pawn:   return U'♟';
        default: return U' ';
    }
}

struct Difficulty { const char* name; int depth; int margin; };
static const Difficulty DIFFS[3] = {
    {"Easy",   2, 70},
    {"Medium", 3, 30},
    {"Hard",   4,  0}
};

class ChessApp {
public:
    ChessApp()
        : window(sf::VideoMode({(unsigned)layout::WIN_W, (unsigned)layout::WIN_H}), "C++ Chess",
                 sf::Style::Titlebar | sf::Style::Close)
        , glyphText(pieceFont, "", 60)
        , uiText(uiFont, "", 18)
        , bigText(uiFont, "", 34)
    {
        window.setFramerateLimit(60);
        if (!pieceFont.openFromFile("C:/Windows/Fonts/seguisym.ttf"))
            throw std::runtime_error("Could not load chess-glyph font (seguisym.ttf)");
        if (!uiFont.openFromFile("C:/Windows/Fonts/segoeui.ttf") &&
            !uiFont.openFromFile("C:/Windows/Fonts/arial.ttf"))
            throw std::runtime_error("Could not load UI font (segoeui.ttf / arial.ttf)");
        state = initialPosition();
    }

    void run() {
        while (window.isOpen()) {
            handleEvents();
            update();
            render();
        }
    }

private:
    sf::RenderWindow window;
    sf::Font pieceFont, uiFont;
    sf::Text glyphText, uiText, bigText;

    Screen screen = Screen::Menu;
    Mode   mode   = Mode::HumanVsHuman;
    Color  humanColor = Color::White;   // in AI mode, the side the human controls
    int    diffIndex  = 1;              // default Medium
    bool   flipped    = false;          // draw board from Black's view

    GameState state;
    std::vector<std::string> history;
    GameEnd ending;
    Move lastMove;

    int  selectedSq = -1;
    bool dragging    = false;
    std::vector<Move> selMoves;         // legal moves from the selected square
    sf::Vector2i mousePix;

    bool promoting = false;
    Move promoBase;                     // from/to of the pending promotion
    std::array<sf::FloatRect, 4> promoRects;
    static constexpr PieceType promoOptions[4] =
        {PieceType::Queen, PieceType::Rook, PieceType::Bishop, PieceType::Knight};

    std::optional<std::future<Move>> aiFuture;
    int gameGen = 0;
    int aiFutureGen = 0;

    // Button rects (recomputed each frame as they are drawn).
    sf::FloatRect btnHvH, btnAIWhite, btnAIBlack, btnDiff[3];
    sf::FloatRect btnNewGame, btnMenu;

    // -------- coordinate transforms --------
    int pixelToSquare(int x, int y) const {
        if (x < 0 || x >= layout::BOARD || y < 0 || y >= layout::BOARD) return -1;
        int dc = x / layout::SQUARE, dr = y / layout::SQUARE;
        int r = flipped ? 7 - dr : dr;
        int c = flipped ? 7 - dc : dc;
        return sqOf(r, c);
    }
    sf::Vector2f squareTopLeft(int sq) const {
        int dr = flipped ? 7 - rowOf(sq) : rowOf(sq);
        int dc = flipped ? 7 - colOf(sq) : colOf(sq);
        return {float(dc * layout::SQUARE), float(dr * layout::SQUARE)};
    }
    sf::Vector2f squareCenter(int sq) const {
        sf::Vector2f tl = squareTopLeft(sq);
        return {tl.x + layout::SQUARE / 2.f, tl.y + layout::SQUARE / 2.f};
    }
    bool humansTurn() const {
        return mode == Mode::HumanVsHuman || state.side == humanColor;
    }

    // -------- game control --------
    void newGame() {
        ++gameGen;
        state = initialPosition();
        history.clear();
        history.push_back(positionKey(state));
        ending = {};
        lastMove = {};
        selectedSq = -1; dragging = false; selMoves.clear();
        promoting = false;
        aiFuture.reset();   // safe: called from the main thread between searches
    }
    void startGame(Mode m, Color human) {
        mode = m;
        humanColor = human;
        flipped = (m == Mode::HumanVsAI && human == Color::Black);
        screen = Screen::Playing;
        newGame();
    }
    void applyMove(const Move& m) {
        state = makeMove(state, m);
        lastMove = m;
        history.push_back(positionKey(state));
        ending = evaluateGameEnd(state, history);
        selectedSq = -1; dragging = false; selMoves.clear();
    }

    // -------- input --------
    void handleEvents() {
        while (const std::optional event = window.pollEvent()) {
            if (event->is<sf::Event::Closed>()) { window.close(); return; }

            if (const auto* mv = event->getIf<sf::Event::MouseMoved>())
                mousePix = mv->position;

            if (const auto* kp = event->getIf<sf::Event::KeyPressed>()) {
                if (kp->code == sf::Keyboard::Key::Escape) {
                    if (promoting) promoting = false;
                    else { screen = Screen::Menu; ++gameGen; }
                }
            }
            if (const auto* mb = event->getIf<sf::Event::MouseButtonPressed>()) {
                if (mb->button == sf::Mouse::Button::Left) {
                    mousePix = mb->position;
                    if (screen == Screen::Menu) onMenuClick(mb->position);
                    else onBoardPress(mb->position);
                }
            }
            if (const auto* mb = event->getIf<sf::Event::MouseButtonReleased>()) {
                if (mb->button == sf::Mouse::Button::Left && screen == Screen::Playing) {
                    mousePix = mb->position;
                    onBoardRelease(mb->position);
                }
            }
        }
    }

    void onMenuClick(sf::Vector2i p) {
        sf::Vector2f f(float(p.x), float(p.y));
        if (btnHvH.contains(f))     startGame(Mode::HumanVsHuman, Color::White);
        if (btnAIWhite.contains(f)) startGame(Mode::HumanVsAI, Color::White);
        if (btnAIBlack.contains(f)) startGame(Mode::HumanVsAI, Color::Black);
        for (int i = 0; i < 3; ++i)
            if (btnDiff[i].contains(f)) diffIndex = i;
    }

    void onBoardPress(sf::Vector2i p) {
        sf::Vector2f f(float(p.x), float(p.y));
        if (btnNewGame.contains(f)) { newGame(); return; }
        if (btnMenu.contains(f))    { screen = Screen::Menu; ++gameGen; return; }

        if (promoting) { onPromoClick(p); return; }
        if (ending.over || !humansTurn() || aiFuture) return;

        int sq = pixelToSquare(p.x, p.y);
        if (sq < 0) return;
        Piece pc = state.board[sq];
        if (!pc.empty() && pc.color == state.side) {
            selectedSq = sq;
            dragging = true;
            selMoves.clear();
            for (const Move& m : generateLegal(state))
                if (m.from == sq) selMoves.push_back(m);
        }
    }

    void onBoardRelease(sf::Vector2i p) {
        if (!dragging) return;
        dragging = false;
        int target = pixelToSquare(p.x, p.y);
        if (target < 0) { selectedSq = -1; selMoves.clear(); return; }

        std::vector<Move> matches;
        for (const Move& m : selMoves) if (m.to == target) matches.push_back(m);
        if (matches.empty()) { selectedSq = -1; selMoves.clear(); return; }

        if (matches[0].promo != PieceType::None) {   // ask which piece
            promoting = true;
            promoBase = matches[0];
        } else {
            applyMove(matches[0]);
        }
    }

    void onPromoClick(sf::Vector2i p) {
        sf::Vector2f f(float(p.x), float(p.y));
        for (int i = 0; i < 4; ++i) {
            if (promoRects[i].contains(f)) {
                Move m = promoBase;
                m.promo = promoOptions[i];
                promoting = false;
                applyMove(m);
                return;
            }
        }
    }

    // -------- AI driver --------
    void update() {
        if (screen != Screen::Playing || ending.over || promoting) return;
        if (mode != Mode::HumanVsAI || state.side == humanColor) return;

        if (!aiFuture) {
            aiFutureGen = gameGen;
            GameState snapshot = state;
            int depth  = DIFFS[diffIndex].depth;
            int margin = DIFFS[diffIndex].margin;
            aiFuture = std::async(std::launch::async,
                                  [snapshot, depth, margin] { return findBestMove(snapshot, depth, margin); });
        }
        if (aiFuture && aiFuture->wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
            Move m = aiFuture->get();
            aiFuture.reset();
            if (aiFutureGen == gameGen && !ending.over && screen == Screen::Playing
                && mode == Mode::HumanVsAI && state.side != humanColor && m.valid())
                applyMove(m);
        }
    }

    // -------- rendering --------
    void render() {
        window.clear(sf::Color(40, 44, 52));
        if (screen == Screen::Menu) drawMenu();
        else { drawBoard(); drawPanel(); if (promoting) drawPromotion(); }
        window.display();
    }

    void drawCentered(sf::Text& t, const sf::String& s, sf::Vector2f center) {
        t.setString(s);
        sf::FloatRect b = t.getLocalBounds();
        t.setOrigin(b.position + b.size / 2.f);
        t.setPosition(center);
        window.draw(t);
    }

    void drawButton(sf::FloatRect r, const sf::String& label, bool hov, bool selected = false) {
        sf::RectangleShape box(r.size);
        box.setPosition(r.position);
        box.setFillColor(selected ? sf::Color(70, 130, 90)
                                   : (hov ? sf::Color(90, 96, 110) : sf::Color(64, 70, 82)));
        box.setOutlineColor(sf::Color(150, 156, 170));
        box.setOutlineThickness(1.5f);
        window.draw(box);
        uiText.setCharacterSize(20);
        uiText.setFillColor(sf::Color::White);
        drawCentered(uiText, label, r.position + r.size / 2.f);
    }

    bool hovered(sf::FloatRect r) const {
        return r.contains(sf::Vector2f(float(mousePix.x), float(mousePix.y)));
    }

    void drawMenu() {
        bigText.setCharacterSize(48);
        bigText.setFillColor(sf::Color::White);
        drawCentered(bigText, "C++ Chess", {layout::WIN_W / 2.f, 90.f});

        float cx = layout::WIN_W / 2.f, w = 320.f, h = 56.f;
        btnHvH     = sf::FloatRect({cx - w / 2, 180.f}, {w, h});
        btnAIWhite = sf::FloatRect({cx - w / 2, 252.f}, {w, h});
        btnAIBlack = sf::FloatRect({cx - w / 2, 324.f}, {w, h});
        drawButton(btnHvH,     "Human vs Human",      hovered(btnHvH));
        drawButton(btnAIWhite, "Play vs AI  (White)", hovered(btnAIWhite));
        drawButton(btnAIBlack, "Play vs AI  (Black)", hovered(btnAIBlack));

        uiText.setCharacterSize(20);
        uiText.setFillColor(sf::Color(200, 204, 214));
        drawCentered(uiText, "AI difficulty", {cx, 420.f});
        float dw = 100.f, gap = 12.f, totalW = 3 * dw + 2 * gap, x0 = cx - totalW / 2;
        for (int i = 0; i < 3; ++i) {
            btnDiff[i] = sf::FloatRect({x0 + i * (dw + gap), 444.f}, {dw, 48.f});
            drawButton(btnDiff[i], DIFFS[i].name, hovered(btnDiff[i]), i == diffIndex);
        }

        uiText.setCharacterSize(16);
        uiText.setFillColor(sf::Color(150, 156, 170));
        drawCentered(uiText, "Drag pieces to move.  Esc returns here.", {cx, layout::WIN_H - 40.f});
    }

    void drawBoard() {
        sf::Color light(240, 217, 181), dark(181, 136, 99);
        sf::RectangleShape sq({float(layout::SQUARE), float(layout::SQUARE)});
        for (int s = 0; s < 64; ++s) {
            sq.setPosition(squareTopLeft(s));
            sq.setFillColor(((rowOf(s) + colOf(s)) % 2 == 0) ? light : dark);
            window.draw(sq);
        }

        if (lastMove.valid())
            for (int s : {lastMove.from, lastMove.to}) {
                sf::RectangleShape hl({float(layout::SQUARE), float(layout::SQUARE)});
                hl.setPosition(squareTopLeft(s));
                hl.setFillColor(sf::Color(246, 246, 105, 120));
                window.draw(hl);
            }
        if (!ending.over && inCheck(state, state.side)) {
            sf::RectangleShape hl({float(layout::SQUARE), float(layout::SQUARE)});
            hl.setPosition(squareTopLeft(findKing(state, state.side)));
            hl.setFillColor(sf::Color(220, 60, 60, 140));
            window.draw(hl);
        }
        if (selectedSq >= 0) {
            sf::RectangleShape hl({float(layout::SQUARE), float(layout::SQUARE)});
            hl.setPosition(squareTopLeft(selectedSq));
            hl.setFillColor(sf::Color(120, 200, 120, 140));
            window.draw(hl);
        }

        uiText.setCharacterSize(13);
        for (int i = 0; i < 8; ++i) {
            int fileCol = flipped ? 7 - i : i;
            int rankRow = flipped ? 7 - i : i;
            uiText.setFillColor((i % 2 == 0) ? sf::Color(181,136,99) : sf::Color(240,217,181));
            uiText.setOrigin({0.f, 0.f});
            uiText.setString(std::string(1, char('a' + fileCol)));
            uiText.setPosition({i * float(layout::SQUARE) + 4.f, layout::BOARD - 18.f});
            window.draw(uiText);
            uiText.setString(std::to_string(8 - rankRow));
            uiText.setPosition({3.f, i * float(layout::SQUARE) + 3.f});
            window.draw(uiText);
        }

        for (const Move& m : selMoves) {
            sf::Vector2f c = squareCenter(m.to);
            if (isCapture(state, m)) {
                sf::CircleShape ring(layout::SQUARE * 0.46f);
                ring.setOrigin({ring.getRadius(), ring.getRadius()});
                ring.setPosition(c);
                ring.setFillColor(sf::Color::Transparent);
                ring.setOutlineThickness(4.f);
                ring.setOutlineColor(sf::Color(40, 40, 40, 150));
                window.draw(ring);
            } else {
                sf::CircleShape dot(layout::SQUARE * 0.14f);
                dot.setOrigin({dot.getRadius(), dot.getRadius()});
                dot.setPosition(c);
                dot.setFillColor(sf::Color(40, 40, 40, 120));
                window.draw(dot);
            }
        }

        for (int s = 0; s < 64; ++s) {
            Piece p = state.board[s];
            if (p.empty() || (dragging && s == selectedSq)) continue;
            drawPiece(p, squareCenter(s));
        }
        if (dragging && selectedSq >= 0)
            drawPiece(state.board[selectedSq], sf::Vector2f(float(mousePix.x), float(mousePix.y)));
    }

    void drawPiece(Piece p, sf::Vector2f center) {
        glyphText.setCharacterSize(60);
        glyphText.setString(sf::String(glyphFor(p.type)));
        if (p.color == Color::White) {
            glyphText.setFillColor(sf::Color(248, 248, 248));
            glyphText.setOutlineColor(sf::Color(30, 30, 30));
        } else {
            glyphText.setFillColor(sf::Color(33, 33, 33));
            glyphText.setOutlineColor(sf::Color(220, 220, 220));
        }
        glyphText.setOutlineThickness(2.f);
        sf::FloatRect b = glyphText.getLocalBounds();
        glyphText.setOrigin(b.position + b.size / 2.f);
        glyphText.setPosition(center);
        window.draw(glyphText);
    }

    void drawPanel() {
        sf::RectangleShape panel({float(layout::PANEL), float(layout::WIN_H)});
        panel.setPosition({float(layout::BOARD), 0.f});
        panel.setFillColor(sf::Color(48, 52, 62));
        window.draw(panel);

        float cx = layout::BOARD + layout::PANEL / 2.f;
        bigText.setCharacterSize(26);
        bigText.setFillColor(sf::Color::White);
        drawCentered(bigText, "C++ Chess", {cx, 34.f});

        uiText.setCharacterSize(17);
        uiText.setFillColor(sf::Color(190, 195, 205));
        std::string modeStr = (mode == Mode::HumanVsHuman)
            ? "Human vs Human"
            : std::string("vs AI (") + DIFFS[diffIndex].name + ")";
        drawCentered(uiText, modeStr, {cx, 72.f});

        std::string status;
        sf::Color statusColor = sf::Color::White;
        if (ending.over) {
            status = ending.message;
            statusColor = sf::Color(245, 222, 120);
        } else {
            status = (state.side == Color::White ? "White to move" : "Black to move");
            if (inCheck(state, state.side)) { status += " — Check!"; statusColor = sf::Color(245, 130, 130); }
        }
        uiText.setCharacterSize(18);
        uiText.setFillColor(statusColor);
        drawCentered(uiText, status, {cx, 134.f});

        if (aiFuture) {
            uiText.setCharacterSize(17);
            uiText.setFillColor(sf::Color(160, 200, 245));
            drawCentered(uiText, "AI is thinking...", {cx, 172.f});
        }

        float bw = layout::PANEL - 40.f, bx = layout::BOARD + 20.f;
        btnNewGame = sf::FloatRect({bx, layout::WIN_H - 130.f}, {bw, 48.f});
        btnMenu    = sf::FloatRect({bx, layout::WIN_H - 72.f},  {bw, 48.f});
        drawButton(btnNewGame, "New Game",  hovered(btnNewGame));
        drawButton(btnMenu,    "Main Menu", hovered(btnMenu));
    }

    void drawPromotion() {
        sf::RectangleShape shade({float(layout::BOARD), float(layout::BOARD)});
        shade.setFillColor(sf::Color(0, 0, 0, 150));
        window.draw(shade);

        float box = layout::SQUARE, totalW = 4 * box;
        float x0 = (layout::BOARD - totalW) / 2.f, y0 = (layout::BOARD - box) / 2.f;
        Color col = state.side;

        uiText.setCharacterSize(16);
        uiText.setFillColor(sf::Color::White);
        drawCentered(uiText, "Promote to:", {layout::BOARD / 2.f, y0 - 26.f});

        for (int i = 0; i < 4; ++i) {
            promoRects[i] = sf::FloatRect({x0 + i * box, y0}, {box, box});
            sf::RectangleShape cell(promoRects[i].size);
            cell.setPosition(promoRects[i].position);
            cell.setFillColor(hovered(promoRects[i]) ? sf::Color(220, 220, 220) : sf::Color(190, 190, 190));
            cell.setOutlineColor(sf::Color(60, 60, 60));
            cell.setOutlineThickness(2.f);
            window.draw(cell);
            drawPiece({promoOptions[i], col},
                      {promoRects[i].position.x + box / 2.f, promoRects[i].position.y + box / 2.f});
        }
    }
};

int main() {
    try {
        ChessApp app;
        app.run();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Fatal: %s\n", e.what());
        return 1;
    }
    return 0;
}
