"""Players for arena evaluation: the learned agent and its fixed opponents.

A player answers choose_moves(boards) for a list of boards where it is to
move. Batch form lets the MCTS player share network batches across games;
the scripted opponents just answer one board at a time.
"""

import random
import subprocess

import chess
import numpy as np

from .mcts import Node, run_simulations, choose_action

_PIECE_VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
                 chess.ROOK: 5, chess.QUEEN: 9}


class RandomPlayer:
    """Uniform random over legal moves: the floor any learning must clear."""

    def __init__(self, seed=0):
        self.rng = random.Random(seed)

    def choose_moves(self, boards):
        return [self.rng.choice(list(b.legal_moves)) for b in boards]


class GreedyMaterialPlayer:
    """Takes the largest immediate material gain, otherwise plays randomly.

    One ply of lookahead and nothing else: it punishes hung pieces, which is
    exactly the skill a young network lacks, so beating it is the first
    meaningful milestone.
    """

    def __init__(self, seed=0):
        self.rng = random.Random(seed)

    def _gain(self, board, move):
        gain = 0
        if board.is_capture(move):
            if board.is_en_passant(move):
                gain += 1
            else:
                victim = board.piece_type_at(move.to_square)
                gain += _PIECE_VALUES.get(victim, 0)
        if move.promotion is not None:
            gain += _PIECE_VALUES.get(move.promotion, 0) - 1
        return gain

    def choose_moves(self, boards):
        out = []
        for b in boards:
            moves = list(b.legal_moves)
            best = max(self._gain(b, m) for m in moves)
            out.append(self.rng.choice(
                [m for m in moves if self._gain(b, m) == best]))
        return out


class EnginePlayer:
    """The repository's C++ alpha-beta engine, spoken to over engine_cli.

    One subprocess serves all queries; each request is a line of
    "<depth> <FEN>" answered with a coordinate move.
    """

    def __init__(self, exe_path, depth):
        self.depth = depth
        self.proc = subprocess.Popen(
            [str(exe_path)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, bufsize=1,
        )

    def choose_moves(self, boards):
        out = []
        for b in boards:
            self.proc.stdin.write("%d %s\n" % (self.depth, b.fen()))
            self.proc.stdin.flush()
            out.append(chess.Move.from_uci(self.proc.stdout.readline().strip()))
        return out

    def close(self):
        if self.proc.poll() is None:
            self.proc.stdin.close()
            self.proc.terminate()


class MCTSPlayer:
    """The learned policy plus search, as used for evaluation matches.

    Fresh tree per move, no Dirichlet noise, and a short burst of tau=1
    sampling at the start of the game so repeated matches do not replay one
    deterministic game.
    """

    def __init__(self, net, device, sims, seed=0, temp_plies=8):
        self.net = net
        self.device = device
        self.sims = sims
        self.temp_plies = temp_plies
        self.rng = np.random.default_rng(seed)

    def choose_moves(self, boards):
        searches = [(b, Node(0.0)) for b in boards]
        run_simulations(searches, self.net, self.device, self.sims)
        out = []
        for b, root in searches:
            tau = 1.0 if b.ply() < self.temp_plies else 0.0
            _, move = choose_action(root, tau, self.rng)
            out.append(move)
        return out
