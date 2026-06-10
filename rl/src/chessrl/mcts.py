"""Monte Carlo tree search with network priors (PUCT), batched across games.

The search follows AlphaGo Zero: selection walks the tree maximizing
Q + U, where U is the prior-weighted exploration bonus, leaves are expanded
with the network's policy as priors and its value as the backup signal, and
no rollouts are played. To keep a GPU busy from Python, many independent
games run their searches in lockstep: each simulation step descends every
active tree, then evaluates all reached leaves in a single batched forward
pass.

Sign convention: a node's value_sum accumulates values from the perspective
of the player to move at that node, so a parent reads its child's mean value
negated.
"""

import math

import chess
import numpy as np

from .encoding import encode_board, legal_actions, encode_move

# Outcome values from the perspective of the player to move.
LOSS, DRAW = -1.0, 0.0


class Node:
    __slots__ = ("prior", "visits", "value_sum", "children")

    def __init__(self, prior):
        self.prior = prior
        self.visits = 0
        self.value_sum = 0.0
        self.children = None      # dict: action index -> (Node, move), or None

    @property
    def expanded(self):
        return self.children is not None

    def q(self):
        return self.value_sum / self.visits if self.visits else 0.0


def select_child(node, c_puct):
    """Pick the child maximizing Q + U from the parent's perspective."""
    sqrt_visits = math.sqrt(node.visits)
    best, best_score = None, -float("inf")
    for action, (child, move) in node.children.items():
        q = -child.q()
        u = c_puct * child.prior * sqrt_visits / (1 + child.visits)
        score = q + u
        if score > best_score:
            best, best_score = (action, child, move), score
    return best


def expand(node, board, logits):
    """Attach children with softmax priors over the legal moves only."""
    moves, indices = legal_actions(board)
    raw = logits[indices]
    raw -= raw.max()
    priors = np.exp(raw)
    priors /= priors.sum()
    node.children = {
        idx: (Node(float(p)), mv) for idx, mv, p in zip(indices, moves, priors)
    }


def terminal_value(board):
    """Value for the side to move if the position is over, else None.

    Inside the tree only automatic rules are checked (checkmate, stalemate,
    insufficient material); clock and repetition draws are adjudicated by the
    game loop, which owns the move history.
    """
    if board.is_checkmate():
        return LOSS
    if board.is_stalemate() or board.is_insufficient_material():
        return DRAW
    return None


def backup(path, value):
    """Propagate a leaf value up the path, flipping perspective each level."""
    for node in reversed(path):
        node.visits += 1
        node.value_sum += value
        value = -value


def add_dirichlet_noise(root, rng, alpha=0.3, eps=0.25):
    actions = list(root.children.keys())
    noise = rng.dirichlet([alpha] * len(actions))
    for a, n in zip(actions, noise):
        child = root.children[a][0]
        child.prior = (1 - eps) * child.prior + eps * float(n)


def run_simulations(searches, net, device, sims, c_puct=1.6,
                    rng=None, root_noise=False):
    """Run `sims` simulations on every (board, root) pair in lockstep.

    `searches` is a list of (chess.Board, Node). Boards are not modified;
    each descent works on a copy. Roots that are unexpanded get a priming
    evaluation first (counted as one simulation), after which optional
    Dirichlet noise is mixed into the root priors.
    """
    from .model import evaluate_batch

    # Prime any unexpanded roots with one batched evaluation.
    fresh = [(b, r) for b, r in searches if not r.expanded]
    if fresh:
        planes = np.stack([encode_board(b) for b, _ in fresh])
        logits, values = evaluate_batch(net, planes, device)
        for (b, r), lo, v in zip(fresh, logits, values):
            expand(r, b, lo)
            backup([r], float(v))
    if root_noise and rng is not None:
        for _, r in searches:
            add_dirichlet_noise(r, rng)

    for _ in range(sims):
        pending = []        # (path, board copy) awaiting net evaluation
        for board, root in searches:
            b = board.copy(stack=False)
            node, path = root, [root]
            while node.expanded:
                action, child, move = select_child(node, c_puct)
                b.push(move)
                node, path = child, path + [child]
            tv = terminal_value(b)
            if tv is not None:
                backup(path, tv)
            else:
                pending.append((path, b))

        if pending:
            planes = np.stack([encode_board(b) for _, b in pending])
            logits, values = evaluate_batch(net, planes, device)
            for (path, b), lo, v in zip(pending, logits, values):
                expand(path[-1], b, lo)
                backup(path, float(v))


def visit_policy(root):
    """Visit counts over root actions, normalized: the training target."""
    actions = np.array(list(root.children.keys()), dtype=np.int64)
    visits = np.array([root.children[a][0].visits for a in actions],
                      dtype=np.float64)
    total = visits.sum()
    if total == 0:
        visits = np.ones_like(visits)
        total = visits.sum()
    return actions, visits / total


def choose_action(root, temperature, rng):
    """Sample an action from visit counts (tau=1) or take the max (tau=0)."""
    actions, probs = visit_policy(root)
    if temperature <= 0.01:
        a = actions[int(np.argmax(probs))]
    else:
        a = rng.choice(actions, p=probs)
    return int(a), root.children[int(a)][1]
