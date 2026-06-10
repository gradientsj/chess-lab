"""AlphaZero-style self-play reinforcement learning for chess.

The package pairs with the C++ engine in the repository root: python-chess
provides the rules on this side (cross-checked against the same published
perft counts the C++ engine is verified with), and the alpha-beta engine is
one of the fixed opponents the learned player is measured against, through
the engine_cli bridge.
"""
