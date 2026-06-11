"""GRPO-style policy-gradient self-play: the second way to learn chess here.

Where the AlphaZero-style loop in train.py distills a search into a network
(the search plays well, the network learns to predict it), this trainer is
pure policy gradient in the style of GRPO (group relative policy
optimization, the algorithm behind DeepSeek-R1's reasoning training and the
GRPO pass in many LLM post-training stacks). The policy plays games by
sampling directly from its own move distribution, no tree search anywhere,
and learns from nothing but game outcomes:

- A group of games is played from the same starting position. Each game's
  reward is its result. Because both colors are played by the same policy,
  every game yields two trajectories (White's moves rewarded with the
  result, Black's with its negation).
- The advantage of each trajectory is its reward standardized within the
  group (subtract the group mean, divide by the group standard deviation),
  GRPO's replacement for a learned value baseline. A group that is all
  draws has zero spread and is skipped: no relative information, no update.
- The update maximizes the PPO-style clipped surrogate using the
  probability ratio between the current policy and the policy that sampled
  the games, for a few epochs per batch, plus an entropy bonus over the
  legal-move distribution. LLM GRPO uses a KL penalty to a reference model
  in place of entropy; training chess from a random initialization has no
  meaningful reference to stay close to, so entropy regularization plays
  the same stabilizing role.

The point of having both trainers in one repository is the comparison:
both are evaluated on the same fixed ladder (random, greedy material, the
C++ alpha-beta engine), so the gap between search-guided self-play and
search-free policy gradient is a measured number rather than folklore.

Run from the rl/ directory:
    python -m chessrl.pg --run pg-baseline
"""

import argparse
import json
import time
from pathlib import Path

import chess
import numpy as np
import torch
import torch.nn.functional as F

from .encoding import encode_board, legal_actions
from .model import PolicyValueNet
from .selfplay import game_outcome
from .train import REPO_ROOT, save_checkpoint, load_net


def sample_moves(net, device, boards, rng, temperature=1.0):
    """Sample a move per board from the masked policy distribution.

    Returns per board: (move, action index, log-prob of the choice, the
    legal action-index list, and the position of the choice in that list).
    """
    planes = np.stack([encode_board(b) for b in boards])
    with torch.inference_mode():
        logits, _ = net(torch.from_numpy(planes).to(device))
    logits = logits.float().cpu().numpy()

    out = []
    for b, row in zip(boards, logits):
        moves, indices = legal_actions(b)
        masked = row[indices] / max(temperature, 1e-3)
        masked -= masked.max()
        probs = np.exp(masked)
        probs /= probs.sum()
        k = int(rng.choice(len(moves), p=probs))
        out.append((moves[k], indices[k], float(np.log(probs[k] + 1e-12)),
                    np.asarray(indices, dtype=np.int16), k))
    return out


def play_group(net, device, n_games, rng, max_plies=200, temperature=1.0):
    """Play a group of self-play games by direct policy sampling.

    Returns a list of trajectories, two per game (one per color), each a
    dict with stacked planes, chosen positions, old log-probs, padded legal
    index lists, and the trajectory reward; plus aggregate game stats.
    """
    games = [{"board": chess.Board(), "steps": [], "z": None}
             for _ in range(n_games)]
    active = games[:]
    while active:
        boards = [g["board"] for g in active]
        choices = sample_moves(net, device, boards, rng, temperature)
        for g, (move, action, logp, legal, k) in zip(active, choices):
            g["steps"].append((encode_board(g["board"]).astype(np.float16),
                               g["board"].turn, legal, k, logp))
            g["board"].push(move)
        still = []
        for g in active:
            z = game_outcome(g["board"], max_plies)
            if z is None:
                still.append(g)
            else:
                g["z"] = z
        active = still

    trajectories = []
    stats = {"white_wins": 0, "draws": 0, "black_wins": 0, "plies": 0}
    for g in games:
        stats["plies"] += g["board"].ply()
        key = ("white_wins" if g["z"] > 0
               else "black_wins" if g["z"] < 0 else "draws")
        stats[key] += 1
        for color in (chess.WHITE, chess.BLACK):
            steps = [s for s in g["steps"] if s[1] == color]
            if not steps:
                continue
            trajectories.append({
                "planes": np.stack([s[0] for s in steps]),
                "legal": [s[2] for s in steps],
                "chosen": np.array([s[3] for s in steps], dtype=np.int64),
                "logp_old": np.array([s[4] for s in steps],
                                     dtype=np.float32),
                "reward": float(g["z"] if color == chess.WHITE else -g["z"]),
            })
    return trajectories, stats


def group_advantages(rewards, eps=1e-6):
    """GRPO's baseline: standardize rewards within the group.

    Returns None when the group has no spread (for example all draws),
    which callers should treat as "nothing to learn from this group".
    """
    r = np.asarray(rewards, dtype=np.float32)
    std = r.std()
    if std < eps:
        return None
    return (r - r.mean()) / std


def masked_log_probs_and_entropy(logits, legal_padded, chosen_pos):
    """Log-prob of each chosen action and entropy, under the legal-move
    softmax. `legal_padded` is (B, L) action indices padded with -1."""
    pad = legal_padded < 0
    safe = legal_padded.clamp(min=0)
    gathered = logits.gather(1, safe)
    gathered = gathered.masked_fill(pad, -1e9)
    logp = F.log_softmax(gathered, dim=1)
    chosen_logp = logp.gather(1, chosen_pos.unsqueeze(1)).squeeze(1)
    probs = logp.exp()
    entropy = -(probs * logp.masked_fill(pad, 0.0)).sum(dim=1)
    return chosen_logp, entropy


def grpo_update(net, optimizer, rows, device, clip_eps, entropy_coef,
                epochs, minibatch):
    """PPO-style clipped surrogate over the collected rows, a few epochs."""
    n = len(rows["chosen"])
    max_legal = max(len(l) for l in rows["legal"])
    legal_padded = np.full((n, max_legal), -1, dtype=np.int64)
    for i, l in enumerate(rows["legal"]):
        legal_padded[i, :len(l)] = l

    planes = torch.from_numpy(rows["planes"].astype(np.float32)).to(device)
    legal_padded = torch.from_numpy(legal_padded).to(device)
    chosen = torch.from_numpy(rows["chosen"]).to(device)
    logp_old = torch.from_numpy(rows["logp_old"]).to(device)
    adv = torch.from_numpy(rows["adv"]).to(device)

    # The network stays in eval mode on purpose: games were sampled under
    # the BatchNorm running statistics, and the probability ratio is only
    # meaningful if the update is computed under the same statistics.
    # Gradients still flow through the BN affine parameters; only the
    # running stats are frozen.
    p_loss_sum = ent_sum = 0.0
    steps = 0
    for _ in range(epochs):
        order = torch.randperm(n, device=device)
        for start in range(0, n, minibatch):
            idx = order[start:start + minibatch]
            logits, _ = net(planes[idx])
            logp_new, entropy = masked_log_probs_and_entropy(
                logits, legal_padded[idx], chosen[idx])
            ratio = (logp_new - logp_old[idx]).exp()
            a = adv[idx]
            surrogate = torch.minimum(
                ratio * a,
                ratio.clamp(1 - clip_eps, 1 + clip_eps) * a)
            loss = -surrogate.mean() - entropy_coef * entropy.mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            p_loss_sum += float(-surrogate.mean())
            ent_sum += float(entropy.mean())
            steps += 1
    return p_loss_sum / max(steps, 1), ent_sum / max(steps, 1)


def evaluate_policy(net, device, cfg, log, iteration, eval_path):
    from .arena import play_match
    from .players import (EnginePlayer, GreedyMaterialPlayer, PolicyPlayer,
                          RandomPlayer)

    opponents = [
        ("random", RandomPlayer(seed=iteration), cfg.eval_games),
        ("greedy", GreedyMaterialPlayer(seed=iteration), cfg.eval_games),
    ]
    exe = Path(cfg.engine_exe)
    if exe.exists():
        opponents += [("engine_d%d" % d, EnginePlayer(exe, d),
                       cfg.eval_games_engine) for d in (1, 2)]

    for name, opp, n in opponents:
        me = PolicyPlayer(net, device, seed=iteration)
        w, d, l = play_match(me, opp, n)
        log("eval vs %-9s  +%d =%d -%d   score %.2f"
            % (name, w, d, l, (w + 0.5 * d) / n))
        with open(eval_path, "a") as f:
            f.write(json.dumps({"iter": iteration, "opponent": name,
                                "wins": w, "draws": d, "losses": l}) + "\n")
        if hasattr(opp, "close"):
            opp.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="pg-baseline")
    ap.add_argument("--iters", type=int, default=100000)
    ap.add_argument("--channels", type=int, default=128)
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--games-per-iter", type=int, default=128)
    ap.add_argument("--group-size", type=int, default=64,
                    help="games per advantage-normalization group")
    ap.add_argument("--max-plies", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--clip-eps", type=float, default=0.2)
    ap.add_argument("--entropy-coef", type=float, default=0.01)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--minibatch", type=int, default=4096)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--eval-games", type=int, default=12)
    ap.add_argument("--eval-games-engine", type=int, default=8)
    ap.add_argument("--engine-exe", default=str(REPO_ROOT / "engine_cli.exe"))
    ap.add_argument("--resume", action="store_true")
    cfg = ap.parse_args()

    run_dir = Path(__file__).resolve().parents[2] / "runs" / cfg.run
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path, eval_path = run_dir / "log.txt", run_dir / "eval.jsonl"

    def log(msg):
        line = "%s  %s" % (time.strftime("%H:%M:%S"), msg)
        print(line, flush=True)
        with open(log_path, "a") as f:
            f.write(line + "\n")

    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(0)

    latest_path = run_dir / "latest.pt"
    start_iter = 0
    if cfg.resume and latest_path.exists():
        net = load_net(latest_path, device)
        ckpt = torch.load(latest_path, map_location="cpu", weights_only=True)
        start_iter = ckpt["iter"] + 1
        log("resumed at iteration %d" % start_iter)
    else:
        net = PolicyValueNet(cfg.channels, cfg.blocks).to(device).eval()
        with open(run_dir / "config.json", "w") as f:
            json.dump(vars(cfg), f, indent=2)

    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    log("device %s, GRPO-style policy gradient, %d games/iter in groups of %d"
        % (device, cfg.games_per_iter, cfg.group_size))

    for it in range(start_iter, cfg.iters):
        t0 = time.time()
        trajectories, stats = [], {"white_wins": 0, "draws": 0,
                                   "black_wins": 0, "plies": 0}
        skipped = 0
        for _ in range(cfg.games_per_iter // cfg.group_size):
            group, gstats = play_group(net, device, cfg.group_size, rng,
                                       cfg.max_plies, cfg.temperature)
            for k in stats:
                stats[k] += gstats[k]
            adv = group_advantages([t["reward"] for t in group])
            if adv is None:
                skipped += 1
                continue
            for t, a in zip(group, adv):
                t["adv"] = a
            trajectories.extend(group)
        n_games = sum(stats[k] for k in
                      ("white_wins", "draws", "black_wins"))
        collect_s = time.time() - t0

        if not trajectories:
            log("iter %d  all groups drawn, nothing to learn (%d games, %.0fs)"
                % (it, n_games, collect_s))
            continue

        rows = {
            "planes": np.concatenate([t["planes"] for t in trajectories]),
            "legal": [l for t in trajectories for l in t["legal"]],
            "chosen": np.concatenate([t["chosen"] for t in trajectories]),
            "logp_old": np.concatenate([t["logp_old"]
                                        for t in trajectories]),
            "adv": np.concatenate([
                np.full(len(t["chosen"]), t["adv"], dtype=np.float32)
                for t in trajectories]),
        }
        t0 = time.time()
        p_loss, entropy = grpo_update(net, optimizer, rows, device,
                                      cfg.clip_eps, cfg.entropy_coef,
                                      cfg.epochs, cfg.minibatch)
        log("iter %d  %d games (%d/%d/%d w/d/b, avg %d plies, %d drawn "
            "groups) %d positions, collect %.0fs train %.0fs  "
            "surrogate %.4f entropy %.2f"
            % (it, n_games, stats["white_wins"], stats["draws"],
               stats["black_wins"], stats["plies"] // max(n_games, 1),
               skipped, len(rows["chosen"]), collect_s, time.time() - t0,
               p_loss, entropy))

        save_checkpoint(latest_path, net, cfg.channels, cfg.blocks, it)
        if it % cfg.eval_every == 0:
            save_checkpoint(run_dir / ("iter_%05d.pt" % it), net,
                            cfg.channels, cfg.blocks, it)
            evaluate_policy(net, device, cfg, log, it, eval_path)


if __name__ == "__main__":
    main()
