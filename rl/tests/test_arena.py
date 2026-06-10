"""Arena mechanics, and one fixed-seed strength ordering: one ply of greed
must outscore uniform random. Seeds make the match deterministic."""

from pathlib import Path

import chess
import pytest

from chessrl.arena import play_match
from chessrl.players import EnginePlayer, GreedyMaterialPlayer, RandomPlayer

ENGINE_EXE = Path(__file__).resolve().parents[2] / "engine_cli.exe"


def test_match_accounting():
    w, d, l = play_match(RandomPlayer(seed=1), RandomPlayer(seed=2),
                         n_games=4, max_plies=60)
    assert w + d + l == 4


def test_greedy_beats_random():
    w, d, l = play_match(GreedyMaterialPlayer(seed=3), RandomPlayer(seed=4),
                         n_games=6, max_plies=160)
    assert (w + 0.5 * d) / 6 > 0.5


@pytest.mark.skipif(not ENGINE_EXE.exists(),
                    reason="engine_cli.exe not built")
def test_engine_bridge_returns_legal_moves():
    eng = EnginePlayer(ENGINE_EXE, depth=1)
    try:
        board = chess.Board()
        (move,) = eng.choose_moves([board])
        assert move in board.legal_moves
    finally:
        eng.close()
