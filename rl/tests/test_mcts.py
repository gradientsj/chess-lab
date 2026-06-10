"""The search must be correct before it can be strong: with an untrained
network, terminal values alone should steer it onto a mate in one."""

import chess
import numpy as np
import torch

from chessrl.mcts import Node, choose_action, run_simulations, visit_policy
from chessrl.model import PolicyValueNet


def _search(fen, sims):
    torch.manual_seed(0)
    net = PolicyValueNet(channels=16, blocks=1).eval()
    board = chess.Board(fen)
    root = Node(0.0)
    run_simulations([(board, root)], net, torch.device("cpu"), sims)
    return board, root


def test_finds_back_rank_mate_in_one():
    board, root = _search("6k1/5ppp/8/8/8/8/8/R6K w - -", sims=128)
    _, move = choose_action(root, temperature=0.0,
                            rng=np.random.default_rng(0))
    assert move == chess.Move.from_uci("a1a8")


def test_visit_policy_is_a_distribution_over_legal_moves():
    board, root = _search(chess.Board().fen(), sims=32)
    actions, probs = visit_policy(root)
    assert abs(probs.sum() - 1.0) < 1e-9
    assert len(actions) == board.legal_moves.count()


def test_simulations_accumulate_visits():
    _, root = _search(chess.Board().fen(), sims=32)
    # priming counts one visit, each simulation one more
    assert root.visits == 33
