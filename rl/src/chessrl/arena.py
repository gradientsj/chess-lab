"""Head-to-head matches between two players, alternating colors.

Games run concurrently: on each step the still-running games are grouped by
which player is to move, and each player answers all of its boards in one
call, so a network-backed player keeps batching its evaluations.
"""

import chess

from .selfplay import game_outcome


def play_match(player_a, player_b, n_games, max_plies=200):
    """Returns (wins, draws, losses) from player_a's perspective.

    Player A takes White in even-numbered games and Black in odd ones, so
    color advantage cancels out over the match.
    """
    games = [{"board": chess.Board(), "a_is_white": i % 2 == 0}
             for i in range(n_games)]
    results = [None] * n_games

    while any(r is None for r in results):
        groups = {True: [], False: []}      # a_to_move -> list of game ids
        for i, g in enumerate(games):
            if results[i] is not None:
                continue
            z = game_outcome(g["board"], max_plies)
            if z is not None:
                results[i] = z
                continue
            white_to_move = g["board"].turn == chess.WHITE
            groups[white_to_move == g["a_is_white"]].append(i)

        for a_to_move, player in ((True, player_a), (False, player_b)):
            ids = groups[a_to_move]
            if not ids:
                continue
            moves = player.choose_moves([games[i]["board"] for i in ids])
            for i, m in zip(ids, moves):
                games[i]["board"].push(m)

    wins = draws = losses = 0
    for g, z in zip(games, results):
        score = z if g["a_is_white"] else -z
        if score > 0:
            wins += 1
        elif score < 0:
            losses += 1
        else:
            draws += 1
    return wins, draws, losses
