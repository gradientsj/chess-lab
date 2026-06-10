"""python-chess provides the rules on the RL side; this cross-checks it
against the same published perft counts the C++ engine is verified with
(see perft.cpp in the repository root), so both halves of the project agree
on what a legal game of chess is."""

import chess
import pytest

CASES = [
    ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -", 4, 197281),
    ("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -", 3, 97862),
    ("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - -", 4, 43238),
    ("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq -", 3, 9467),
    ("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ -", 2, 1486),
]


def perft(board, depth):
    if depth == 0:
        return 1
    return sum(perft_after(board, m, depth) for m in board.legal_moves)


def perft_after(board, move, depth):
    board.push(move)
    n = perft(board, depth - 1)
    board.pop()
    return n


@pytest.mark.parametrize("fen,depth,expected", CASES)
def test_perft_matches_published_counts(fen, depth, expected):
    assert perft(chess.Board(fen), depth) == expected
