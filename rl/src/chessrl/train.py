"""The training loop: self-play, gradient steps, gating, evaluation.

Each iteration follows AlphaGo Zero's recipe at desktop scale. Worker
processes play a batch of self-play games with the current best network and
ship the samples back; the trainer takes gradient steps on a replay buffer
of recent positions; the freshly trained candidate then has to beat the
current best in a gating match before it becomes the network that generates
the next round of games. Every few iterations the best network is measured
against fixed opponents (random, greedy material, and the repository's
alpha-beta engine over the CLI bridge) so progress is a curve against
unmoving yardsticks rather than a feeling.

Run from the rl/ directory, for example:
    python -m chessrl.train --run baseline --games-per-iter 192 --workers 6
"""

import argparse
import collections
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .encoding import POLICY_SIZE
from .model import PolicyValueNet

REPO_ROOT = Path(__file__).resolve().parents[3]


def _worker_play(args):
    """Self-play task body, executed in a spawned worker process."""
    (ckpt_path, n_games, sims, parallel, temp_plies, max_plies, seed) = args
    import torch as t
    from .selfplay import play_games

    t.set_num_threads(2)
    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    ckpt = t.load(ckpt_path, map_location=device, weights_only=True)
    net = PolicyValueNet(ckpt["channels"], ckpt["blocks"]).to(device).eval()
    net.load_state_dict(ckpt["model"])

    rng = np.random.default_rng(seed)
    return play_games(net, device, n_games, sims, rng,
                      max_plies=max_plies, temp_plies=temp_plies,
                      parallel=parallel)


def save_checkpoint(path, net, channels, blocks, iteration):
    torch.save({"model": net.state_dict(), "channels": channels,
                "blocks": blocks, "iter": iteration}, path)


def load_net(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=True)
    net = PolicyValueNet(ckpt["channels"], ckpt["blocks"]).to(device).eval()
    net.load_state_dict(ckpt["model"])
    return net


def train_steps(net, optimizer, buffer, steps, batch_size, device, rng):
    """Gradient steps on uniformly sampled buffer positions."""
    net.train()
    p_loss_sum = v_loss_sum = 0.0
    for _ in range(steps):
        idx = rng.integers(0, len(buffer), size=batch_size)
        planes = np.stack([buffer[i][0] for i in idx]).astype(np.float32)
        target_pi = np.zeros((batch_size, POLICY_SIZE), dtype=np.float32)
        for row, i in enumerate(idx):
            target_pi[row, buffer[i][1].astype(np.int64)] = buffer[i][2]
        target_z = np.array([buffer[i][3] for i in idx], dtype=np.float32)

        x = torch.from_numpy(planes).to(device)
        pi = torch.from_numpy(target_pi).to(device)
        z = torch.from_numpy(target_z).to(device)

        logits, v = net(x)
        policy_loss = -(pi * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
        value_loss = F.mse_loss(v.squeeze(1), z)
        loss = policy_loss + value_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        p_loss_sum += float(policy_loss)
        v_loss_sum += float(value_loss)
    net.eval()
    return p_loss_sum / steps, v_loss_sum / steps


def evaluate_vs_fixed(net, device, cfg, log, iteration, eval_path):
    """Match the network against the fixed opponents and record the results."""
    from .arena import play_match
    from .players import (EnginePlayer, GreedyMaterialPlayer, MCTSPlayer,
                          RandomPlayer)

    opponents = [
        ("random", RandomPlayer(seed=iteration), cfg.eval_games),
        ("greedy", GreedyMaterialPlayer(seed=iteration), cfg.eval_games),
    ]
    exe = Path(cfg.engine_exe)
    if exe.exists():
        opponents += [("engine_d%d" % d,
                       EnginePlayer(exe, d), cfg.eval_games_engine)
                      for d in (1, 2)]
    else:
        log("engine_cli not found at %s, skipping engine eval" % exe)

    for name, opp, n in opponents:
        me = MCTSPlayer(net, device, cfg.eval_sims, seed=iteration)
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
    ap.add_argument("--run", default="baseline")
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--channels", type=int, default=128)
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--games-per-iter", type=int, default=192)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--parallel", type=int, default=32,
                    help="lockstep games per worker")
    ap.add_argument("--sims", type=int, default=96)
    ap.add_argument("--temp-plies", type=int, default=30)
    ap.add_argument("--max-plies", type=int, default=200)
    ap.add_argument("--buffer", type=int, default=250000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--gate-games", type=int, default=16)
    ap.add_argument("--gate-sims", type=int, default=48)
    ap.add_argument("--gate-threshold", type=float, default=0.55)
    ap.add_argument("--eval-every", type=int, default=2)
    ap.add_argument("--eval-sims", type=int, default=64)
    ap.add_argument("--eval-games", type=int, default=12)
    ap.add_argument("--eval-games-engine", type=int, default=8)
    ap.add_argument("--engine-exe",
                    default=str(REPO_ROOT / "engine_cli.exe"))
    ap.add_argument("--resume", action="store_true")
    cfg = ap.parse_args()

    run_dir = Path(__file__).resolve().parents[2] / "runs" / cfg.run
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "log.txt"
    eval_path = run_dir / "eval.jsonl"

    def log(msg):
        line = "%s  %s" % (time.strftime("%H:%M:%S"), msg)
        print(line, flush=True)
        with open(log_path, "a") as f:
            f.write(line + "\n")

    torch.set_num_threads(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(0)

    best_path = run_dir / "best.pt"
    start_iter = 0
    if cfg.resume and best_path.exists():
        net = load_net(best_path, device)
        ckpt = torch.load(best_path, map_location="cpu", weights_only=True)
        start_iter = ckpt["iter"] + 1
        log("resumed from %s at iteration %d" % (best_path, start_iter))
    else:
        net = PolicyValueNet(cfg.channels, cfg.blocks).to(device).eval()
        save_checkpoint(best_path, net, cfg.channels, cfg.blocks, -1)
        with open(run_dir / "config.json", "w") as f:
            json.dump(vars(cfg), f, indent=2)

    n_params = sum(p.numel() for p in net.parameters())
    log("device %s, %.2fM parameters, %d workers x %d lockstep games"
        % (device, n_params / 1e6, cfg.workers, cfg.parallel))

    optimizer = torch.optim.SGD(net.parameters(), lr=cfg.lr, momentum=0.9,
                                weight_decay=1e-4)
    buffer = collections.deque(maxlen=cfg.buffer)

    with ProcessPoolExecutor(max_workers=cfg.workers) as pool:
        for it in range(start_iter, cfg.iters):
            # --- self-play with the current best network -----------------
            t0 = time.time()
            per_worker = cfg.games_per_iter // cfg.workers
            tasks = [(str(best_path), per_worker, cfg.sims, cfg.parallel,
                      cfg.temp_plies, cfg.max_plies, it * cfg.workers + w)
                     for w in range(cfg.workers)]
            games = {"white_wins": 0, "draws": 0, "black_wins": 0, "plies": 0}
            new_samples = 0
            for samples, stats in pool.map(_worker_play, tasks):
                buffer.extend(samples)
                new_samples += len(samples)
                for k in games:
                    games[k] += stats[k]
            n_games = games["white_wins"] + games["draws"] + games["black_wins"]
            log("iter %d  self-play %d games (%d/%d/%d w/d/b, avg %d plies) "
                "%d samples in %.0fs, buffer %d"
                % (it, n_games, games["white_wins"], games["draws"],
                   games["black_wins"], games["plies"] // max(n_games, 1),
                   new_samples, time.time() - t0, len(buffer)))

            # --- gradient steps on the replay buffer ---------------------
            t0 = time.time()
            steps = max(200, 2 * new_samples // cfg.batch)
            candidate = load_net(best_path, device)
            candidate.load_state_dict(net.state_dict())
            opt_state = optimizer.state_dict()
            optimizer = torch.optim.SGD(candidate.parameters(), lr=cfg.lr,
                                        momentum=0.9, weight_decay=1e-4)
            optimizer.load_state_dict(opt_state)
            p_loss, v_loss = train_steps(candidate, optimizer, buffer, steps,
                                         cfg.batch, device, rng)
            log("iter %d  trained %d steps in %.0fs  policy loss %.3f  "
                "value loss %.3f" % (it, steps, time.time() - t0,
                                     p_loss, v_loss))

            # --- gating: the candidate must beat the best -----------------
            from .arena import play_match
            from .players import MCTSPlayer
            t0 = time.time()
            w, d, l = play_match(
                MCTSPlayer(candidate, device, cfg.gate_sims, seed=2 * it),
                MCTSPlayer(net, device, cfg.gate_sims, seed=2 * it + 1),
                cfg.gate_games)
            score = (w + 0.5 * d) / cfg.gate_games
            promoted = score >= cfg.gate_threshold
            log("iter %d  gate +%d =%d -%d  score %.2f  %s  (%.0fs)"
                % (it, w, d, l, score,
                   "promoted" if promoted else "rejected",
                   time.time() - t0))
            if promoted:
                net = candidate
                save_checkpoint(best_path, net, cfg.channels, cfg.blocks, it)
            save_checkpoint(run_dir / ("iter_%04d.pt" % it), candidate,
                            cfg.channels, cfg.blocks, it)

            # --- fixed-opponent evaluation -------------------------------
            if it % cfg.eval_every == 0:
                evaluate_vs_fixed(net, device, cfg, log, it, eval_path)


if __name__ == "__main__":
    main()
