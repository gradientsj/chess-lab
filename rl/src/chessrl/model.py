"""The policy-value network: a small residual tower with two heads.

This is the AlphaZero shape scaled down to a single consumer GPU: a
convolutional stem, a stack of residual blocks, a policy head that emits one
logit per (move type, from square) pair, and a value head squashed to
[-1, 1]. At the default 128 channels and 6 blocks it is about 1.9M
parameters.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoding import N_PLANES, POLICY_SIZE


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.c1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(channels)
        self.c2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return F.relu(x + y)


class PolicyValueNet(nn.Module):
    def __init__(self, channels=128, blocks=6):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(N_PLANES, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.tower = nn.Sequential(*[ResBlock(channels) for _ in range(blocks)])

        # One logit per move-type plane per from-square; flattening (73, 8, 8)
        # in C-order matches encoding.move_to_index (plane * 64 + from_square).
        self.policy = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 73, 1),
        )
        self.value = nn.Sequential(
            nn.Conv2d(channels, 8, 1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
        )
        self.value_fc = nn.Sequential(
            nn.Linear(8 * 64, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        x = self.tower(self.stem(x))
        logits = self.policy(x).flatten(1)            # (B, 4672)
        v = self.value_fc(self.value(x).flatten(1))   # (B, 1)
        return logits, torch.tanh(v)


@torch.inference_mode()
def evaluate_batch(net, planes, device):
    """Run the net on a stacked (B, 19, 8, 8) numpy batch.

    Returns (logits, values) as numpy arrays on the CPU.
    """
    x = torch.from_numpy(planes).to(device)
    if device.type == "cuda":
        with torch.autocast("cuda"):
            logits, v = net(x)
    else:
        logits, v = net(x)
    return logits.float().cpu().numpy(), v.float().cpu().numpy().reshape(-1)
