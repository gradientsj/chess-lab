"""Every legal move in any position must round-trip through the action
space, and the board planes must describe the position the mover sees."""

import random

import chess
import numpy as np

from chessrl.encoding import (N_PLANES, POLICY_SIZE, encode_board,
                              encode_move, index_to_move, legal_actions,
                              mirror_move)


def random_positions(n_games=12, max_plies=80, seed=7):
    rng = random.Random(seed)
    boards = []
    for _ in range(n_games):
        b = chess.Board()
        for _ in range(rng.randrange(max_plies)):
            moves = list(b.legal_moves)
            if not moves or b.is_game_over():
                break
            b.push(rng.choice(moves))
        boards.append(b)
    return boards


def test_every_legal_move_round_trips():
    for board in random_positions():
        mover_frame = board if board.turn == chess.WHITE else board.mirror()
        for move in board.legal_moves:
            idx = encode_move(move, board.turn)
            assert 0 <= idx < POLICY_SIZE
            decoded = index_to_move(idx, mover_frame)
            if board.turn == chess.BLACK:
                decoded = mirror_move(decoded)
            assert decoded == move, (board.fen(), move.uci(), decoded.uci())


def test_indices_are_unique_per_position():
    for board in random_positions():
        _, indices = legal_actions(board)
        assert len(indices) == len(set(indices))


def test_underpromotions_have_their_own_actions():
    board = chess.Board("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ -")
    promos = [m for m in board.legal_moves if m.promotion is not None]
    assert len({encode_move(m, board.turn) for m in promos}) == len(promos)


def test_start_position_planes():
    planes = encode_board(chess.Board())
    assert planes.shape == (N_PLANES, 8, 8)
    assert planes[:12].sum() == 32                       # all pieces present
    assert planes[0].sum() == 8                          # my pawns
    assert (planes[0][1] == 1).all()                     # on my second rank
    assert planes[12:16].sum() == 4 * 64                 # all castling rights
    assert (planes[18] == 1).all()                       # ones plane


def test_black_sees_a_mirrored_board():
    board = chess.Board()
    board.push_uci("e2e4")
    planes = encode_board(board)                         # Black to move
    # Black's own pawns appear on rank index 1, just as White's do for White.
    assert planes[0][1].sum() == 8
    # The white pawn on e4 appears as an enemy pawn on the mirrored e5.
    assert planes[6, chess.square_rank(chess.square_mirror(chess.E4)),
                  chess.square_file(chess.E4)] == 1
