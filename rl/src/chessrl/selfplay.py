"""Self-play: the network plays itself and every position becomes a sample.

Each completed game yields one training sample per position: the encoded
board, the search's visit distribution as the policy target (stored sparse),
and the final game result from that position's side to move as the value
target. Games run in lockstep so the network evaluates leaves for all of
them in shared batches.
"""

import numpy as np
import chess

from .encoding import encode_board
from .mcts import Node, run_simulations, visit_policy, choose_action


def game_outcome(board, max_plies, adjudicate_margin=5):
    """White's score (1, 0, -1) if the game is over under full rules.

    The game loop owns the move history, so clock and repetition draws are
    decided here rather than inside the tree. Games that hit the ply cap are
    adjudicated by material: a lead of at least `adjudicate_margin` pawns of
    material counts as a win, anything closer is a draw.
    """
    outcome = board.outcome()
    if outcome is not None:
        if outcome.winner is None:
            return 0
        return 1 if outcome.winner == chess.WHITE else -1
    if board.halfmove_clock >= 100 or board.is_repetition(3):
        return 0
    if board.ply() >= max_plies:
        values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
                  chess.ROOK: 5, chess.QUEEN: 9}
        diff = sum(values.get(p.piece_type, 0) * (1 if p.color else -1)
                   for p in board.piece_map().values())
        if diff >= adjudicate_margin:
            return 1
        if diff <= -adjudicate_margin:
            return -1
        return 0
    return None


class _Game:
    __slots__ = ("board", "root", "samples")

    def __init__(self):
        self.board = chess.Board()
        self.root = Node(0.0)
        self.samples = []     # (planes f16, actions i16, probs f16, turn)


def play_games(net, device, n_games, sims, rng, max_plies=200,
               temp_plies=30, parallel=64):
    """Play `n_games` of self-play and return (samples, stats).

    Samples are tuples (planes, actions, probs, z) with the policy target
    sparse over the visited root actions. Stats counts wins, draws, losses
    from White's perspective and total plies.
    """
    queue = n_games
    active = []
    finished_samples = []
    stats = {"white_wins": 0, "draws": 0, "black_wins": 0, "plies": 0}

    while queue > 0 or active:
        while queue > 0 and len(active) < parallel:
            active.append(_Game())
            queue -= 1

        run_simulations([(g.board, g.root) for g in active], net, device,
                        sims, rng=rng, root_noise=True)

        still = []
        for g in active:
            actions, probs = visit_policy(g.root)
            g.samples.append((
                encode_board(g.board).astype(np.float16),
                actions.astype(np.int16),
                probs.astype(np.float16),
                g.board.turn,
            ))
            tau = 1.0 if g.board.ply() < temp_plies else 0.0
            action, move = choose_action(g.root, tau, rng)
            g.board.push(move)
            g.root = g.root.children[action][0]

            z = game_outcome(g.board, max_plies)
            if z is None:
                still.append(g)
                continue
            stats["plies"] += g.board.ply()
            if z > 0:
                stats["white_wins"] += 1
            elif z < 0:
                stats["black_wins"] += 1
            else:
                stats["draws"] += 1
            for planes, acts, ps, turn in g.samples:
                zz = z if turn == chess.WHITE else -z
                finished_samples.append((planes, acts, ps, np.int8(zz)))
        active = still

    return finished_samples, stats
