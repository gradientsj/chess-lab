# Chess Lab

A chess project in two halves that share one standard of correctness.

The first half is a desktop chess game in C++ with SFML 3: human vs human or
human vs AI, every rule of the game, mouse drag-and-drop. The engine lives in
a single UI-independent header and its move generator is verified against
published perft counts, more than ten million positions across five standard
test suites.

The second half, in `rl/`, is an AlphaZero-style reinforcement learning lab
in Python: a policy-value network that learns chess from self-play alone,
guided by Monte Carlo tree search, trained on a single consumer GPU, and
measured against fixed opponents that include the C++ engine itself through
a small CLI bridge. The rules library on the Python side is cross-checked
against the same published perft counts the C++ engine is verified with, so
both halves agree on what a legal game of chess is.

![board](shot_board.png)

## The game (C++)

- **Full legal rules**: castling, en passant, promotion (with a picker),
  check, checkmate, stalemate, the 50-move rule, threefold repetition, and
  insufficient-material draws.
- **Two modes**: Human vs Human, or Human vs AI (play as White or Black).
- **Search-based AI**: negamax with alpha-beta pruning, a quiescence search,
  and a material + piece-square evaluation. Three difficulties (Easy /
  Medium / Hard) set the search depth; the AI runs on a background thread so
  the window stays responsive.
- **UI**: drag-and-drop, legal-move dots, last-move and check highlights,
  board flips when you play Black.

### Build & run (Windows)

Requires the MSYS2 mingw64 toolchain with SFML 3, and the Windows fonts
`seguisym.ttf` (chess glyphs) and `segoeui.ttf` (UI text), which ship with
Windows.

```powershell
.\build.ps1        # compile -> chess.exe  (add -console to keep a console)
.\run.ps1          # run (puts the SFML DLLs on PATH for you)
```

Manual build:

```powershell
$env:PATH = "C:\msys64\mingw64\bin;$env:PATH"
g++ -std=c++20 -O2 main.cpp -o chess.exe -lsfml-graphics -lsfml-window -lsfml-system -mwindows
```

### Engine tests

```powershell
g++ -std=c++20 -O2 perft.cpp   -o perft.exe   ; .\perft.exe
g++ -std=c++20 -O2 ai_test.cpp -o ai_test.exe ; .\ai_test.exe
```

`perft.exe` counts legal move sequences from five standard positions and
compares them against published reference values (10.4M positions, all
matching), which pins down castling, en passant, promotion, and check
handling exactly. `ai_test.exe` checks that the search finds a mate-in-one,
wins a hanging queen, and that a self-play game terminates.

## The learning half (rl/)

`rl/` implements the AlphaGo Zero recipe at desktop scale. Nothing about
chess is told to the network beyond the rules: no opening book, no
hand-written evaluation, no human games. It learns entirely from games it
plays against itself.

- **Encoding** (`encoding.py`): positions become 19 feature planes from the
  side to move's perspective; moves use the AlphaZero 8x8x73 action space
  (56 queen-style moves, 8 knight moves, 9 underpromotions per from-square).
- **Network** (`model.py`): a residual tower with policy and value heads,
  about 1.9M parameters at the default 128 channels x 6 blocks.
- **Search** (`mcts.py`): PUCT tree search using the network's policy as
  priors and its value head instead of rollouts. Many games run their
  searches in lockstep so leaf evaluations batch onto the GPU.
- **Self-play** (`selfplay.py`): every position of every game becomes a
  training sample; the search's visit distribution is the policy target and
  the final result is the value target. Dirichlet noise at the root and
  temperature sampling in the opening keep the data varied.
- **Training** (`train.py`): worker processes generate games with the
  current best network; the trainer takes SGD steps on a replay buffer of
  recent positions; a candidate is promoted only after winning a gating
  match against the current best, as in AlphaGo Zero.
- **Measurement** (`arena.py`, `players.py`): the trained network is
  evaluated on a fixed ladder: a uniform random player, a one-ply greedy
  material player, and the repository's alpha-beta engine at increasing
  depths, spoken to over `engine_cli`. Results land in
  `runs/<name>/eval.jsonl`, so progress is a curve against unmoving
  yardsticks.

### Two ways to learn, measured against each other

The repository trains chess two different ways on purpose.

`train.py` is **search-guided self-play** in the AlphaGo Zero family,
sometimes called expert iteration: the tree search plays better than the
raw network, the network learns to predict the search, and the improved
network makes the search stronger on the next iteration. The search is the
teacher.

`pg.py` is **pure policy gradient** in the style of GRPO (group relative
policy optimization, the algorithm behind DeepSeek-R1's reasoning training
and a standard pass in LLM post-training stacks): the policy samples whole
games directly from its own move distribution, with no search anywhere,
and learns from nothing but outcomes. Rewards are standardized within a
group of games, GRPO's substitute for a learned value baseline, and the
update is the PPO-style clipped surrogate. Where LLM GRPO penalizes KL
divergence against a reference model, a from-scratch chess policy has no
reference worth staying close to, so an entropy bonus fills the same
stabilizing role. One implementation detail worth knowing: updates run
under the same BatchNorm statistics that sampled the games, because the
probability ratio at the heart of the method stops meaning anything if the
normalization shifts between sampling and update.

Both trainers share the encoding, the network architecture, and the
evaluation ladder, so the question "how much does search-guided learning
buy over pure policy gradient at equal parameters" gets a measured answer
rather than an assumed one. Methods that fit chess poorly at this scale
are left out deliberately: value-based control in the DQN family struggles
with a 4,672-action space and a reward that only arrives at game end, and
classic actor-critic with a learned baseline is already half-covered by
the value head the search-guided loop trains.

### Run it

```bash
cd rl
pip install -e ".[dev]"          # python-chess, numpy, torch
python -m pytest tests -q        # CPU-only tests
python -m chessrl.train --run baseline    # search-guided self-play
python -m chessrl.pg --run pg-baseline    # GRPO-style policy gradient
```

The tests cross-check python-chess against the same published perft counts
as the C++ engine, verify that every legal move in random positions
round-trips through the 4672-action encoding, and confirm the search finds
a mate-in-one with an untrained network, which isolates the correctness of
the tree from the strength of the model.

The engine bridge is built with `g++ -std=c++20 -O2 engine_cli.cpp -o
engine_cli.exe` (add `-static` on Windows so the executable does not depend
on the MinGW runtime DLLs). If the executable is absent, evaluation simply
skips the engine rungs of the ladder.

## Project layout

| Path | Purpose |
|------|---------|
| `chess.hpp`       | Engine: board, rules, move generation, evaluation, search (UI-independent). |
| `main.cpp`        | SFML front-end: window, menu, board rendering, input. |
| `perft.cpp`       | Move-generation verification against published counts. |
| `ai_test.cpp`     | Evaluation and search sanity checks. |
| `engine_cli.cpp`  | Line-based bridge: `<depth> <FEN>` in, best move out. Used by the RL arena. |
| `rl/src/chessrl/` | The self-play learning package. |
| `rl/tests/`       | CPU-only tests for encoding, search, arena, and rules. |
