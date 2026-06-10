import numpy as np
import torch

from chessrl.encoding import N_PLANES, POLICY_SIZE
from chessrl.model import PolicyValueNet, evaluate_batch


def test_forward_shapes_and_value_range():
    net = PolicyValueNet(channels=32, blocks=2).eval()
    x = torch.randn(5, N_PLANES, 8, 8)
    logits, v = net(x)
    assert logits.shape == (5, POLICY_SIZE)
    assert v.shape == (5, 1)
    assert (v.abs() <= 1).all()
    assert torch.isfinite(logits).all()


def test_evaluate_batch_numpy_interface():
    net = PolicyValueNet(channels=32, blocks=2).eval()
    planes = np.random.rand(3, N_PLANES, 8, 8).astype(np.float32)
    logits, values = evaluate_batch(net, planes, torch.device("cpu"))
    assert logits.shape == (3, POLICY_SIZE)
    assert values.shape == (3,)
