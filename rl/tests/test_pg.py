"""The GRPO pieces: group-relative advantages, masked log-probs, and one
update step that moves probability toward the better trajectory."""

import chess
import numpy as np
import torch

from chessrl.model import PolicyValueNet
from chessrl.pg import (group_advantages, grpo_update,
                        masked_log_probs_and_entropy, play_group)


def test_group_advantages_standardize():
    adv = group_advantages([1.0, -1.0])
    assert adv is not None
    assert abs(adv.mean()) < 1e-6
    assert adv[0] > 0 > adv[1]


def test_all_draw_group_is_skipped():
    assert group_advantages([0.0, 0.0, 0.0, 0.0]) is None


def test_masked_log_probs_ignore_padding():
    logits = torch.zeros(1, 10)
    logits[0, 3] = 5.0
    legal = torch.tensor([[3, 7, -1, -1]])
    chosen = torch.tensor([0])               # position of action 3
    logp, entropy = masked_log_probs_and_entropy(logits, legal, chosen)
    # two legal moves, one with a much larger logit
    assert logp.exp().item() > 0.95
    assert 0.0 <= entropy.item() < 0.3


def test_play_group_yields_two_sided_trajectories():
    torch.manual_seed(0)
    net = PolicyValueNet(channels=16, blocks=1).eval()
    rng = np.random.default_rng(0)
    trajectories, stats = play_group(net, torch.device("cpu"), n_games=2,
                                     rng=rng, max_plies=40)
    n_games = stats["white_wins"] + stats["draws"] + stats["black_wins"]
    assert n_games == 2
    assert 2 <= len(trajectories) <= 4
    for t in trajectories:
        assert len(t["planes"]) == len(t["chosen"]) == len(t["logp_old"])
        assert t["reward"] in (-1.0, 0.0, 1.0)


def test_update_moves_probability_toward_positive_advantage():
    torch.manual_seed(0)
    net = PolicyValueNet(channels=16, blocks=1).eval()
    device = torch.device("cpu")
    board = chess.Board()

    from chessrl.encoding import encode_board, legal_actions
    moves, indices = legal_actions(board)
    planes = encode_board(board)

    def prob_of(action_pos):
        with torch.inference_mode():
            logits, _ = net(torch.from_numpy(planes[None].astype(np.float32)))
        row = logits[0].numpy()[indices]
        row -= row.max()
        p = np.exp(row)
        return (p / p.sum())[action_pos]

    before = prob_of(0)
    rows = {
        "planes": np.stack([planes, planes]),
        "legal": [np.asarray(indices, dtype=np.int16)] * 2,
        "chosen": np.array([0, 1], dtype=np.int64),
        "logp_old": np.log(np.array([before, before], dtype=np.float32)),
        "adv": np.array([1.0, -1.0], dtype=np.float32),
    }
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    grpo_update(net, optimizer, rows, device, clip_eps=0.2,
                entropy_coef=0.0, epochs=3, minibatch=2)
    assert prob_of(0) > before
