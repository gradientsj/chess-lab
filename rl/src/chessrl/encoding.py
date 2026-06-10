"""Board and move encoding for the policy-value network.

The board is always encoded from the side to move's perspective: when Black
is to move, ranks are mirrored and piece colors are swapped, so the network
always sees itself as the player pushing up the board. Moves are encoded the
same way, in the mover's frame.

Moves use the AlphaZero 8x8x73 action space. For each from-square there are
56 queen-style moves (8 directions times up to 7 squares), 8 knight moves,
and 9 underpromotions (3 directions times knight/bishop/rook). Promotions to
a queen are encoded as ordinary queen-style moves onto the last rank. The
flattened index is plane * 64 + from_square, which matches a (73, 8, 8)
policy head flattened in C-order.
"""

import chess
import numpy as np

N_PLANES = 19
POLICY_SIZE = 73 * 64

# Directions as (rank delta, file delta) in the mover's frame.
QUEEN_DIRS = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
KNIGHT_DIRS = [(2, 1), (1, 2), (-1, 2), (-2, 1), (-2, -1), (-1, -2), (1, -2), (2, -1)]
UNDERPROMOS = [chess.KNIGHT, chess.BISHOP, chess.ROOK]

_QUEEN_DIR_INDEX = {d: i for i, d in enumerate(QUEEN_DIRS)}
_KNIGHT_DIR_INDEX = {d: i for i, d in enumerate(KNIGHT_DIRS)}


def mirror_move(move):
    """The same move seen from the other player's frame (ranks flipped)."""
    return chess.Move(
        chess.square_mirror(move.from_square),
        chess.square_mirror(move.to_square),
        promotion=move.promotion,
    )


def move_to_index(move):
    """Action index for a move already expressed in the mover's frame."""
    fr, ff = chess.square_rank(move.from_square), chess.square_file(move.from_square)
    tr, tf = chess.square_rank(move.to_square), chess.square_file(move.to_square)
    dr, df = tr - fr, tf - ff

    if move.promotion is not None and move.promotion != chess.QUEEN:
        plane = 64 + (df + 1) * 3 + UNDERPROMOS.index(move.promotion)
    elif (dr, df) in _KNIGHT_DIR_INDEX:
        plane = 56 + _KNIGHT_DIR_INDEX[(dr, df)]
    else:
        dist = max(abs(dr), abs(df))
        direction = (dr // dist if dr else 0, df // dist if df else 0)
        plane = _QUEEN_DIR_INDEX[direction] * 7 + (dist - 1)
    return plane * 64 + move.from_square


def index_to_move(index, board_mover_frame):
    """Inverse of move_to_index. Needs the board to add queen promotions."""
    plane, from_sq = divmod(index, 64)
    fr, ff = chess.square_rank(from_sq), chess.square_file(from_sq)

    if plane >= 64:
        u = plane - 64
        df, promo = u // 3 - 1, UNDERPROMOS[u % 3]
        return chess.Move(from_sq, chess.square(ff + df, fr + 1), promotion=promo)
    if plane >= 56:
        dr, df = KNIGHT_DIRS[plane - 56]
        return chess.Move(from_sq, chess.square(ff + df, fr + dr))

    dr, df = QUEEN_DIRS[plane // 7]
    dist = plane % 7 + 1
    to_sq = chess.square(ff + df * dist, fr + dr * dist)
    promo = None
    piece = board_mover_frame.piece_type_at(from_sq)
    if piece == chess.PAWN and chess.square_rank(to_sq) == 7:
        promo = chess.QUEEN
    return chess.Move(from_sq, to_sq, promotion=promo)


def encode_move(move, turn):
    """Action index for a move on a real board with `turn` to play."""
    if turn == chess.BLACK:
        move = mirror_move(move)
    return move_to_index(move)


def legal_actions(board):
    """Aligned lists: legal moves (real frame) and their action indices."""
    moves = list(board.legal_moves)
    indices = [encode_move(m, board.turn) for m in moves]
    return moves, indices


def encode_board(board):
    """A (19, 8, 8) float32 tensor of the position, mover's perspective.

    Planes 0-5 are the mover's pawns, knights, bishops, rooks, queens, king;
    6-11 the opponent's in the same order; 12-15 castling rights (mover
    kingside, mover queenside, opponent kingside, opponent queenside); 16 the
    en passant target square; 17 the halfmove clock scaled to [0, 1]; 18 is
    all ones so convolutions can see the board edge.
    """
    planes = np.zeros((N_PLANES, 8, 8), dtype=np.float32)
    mover = board.turn
    flip = mover == chess.BLACK

    for sq, piece in board.piece_map().items():
        if flip:
            sq = chess.square_mirror(sq)
        base = 0 if piece.color == mover else 6
        planes[base + piece.piece_type - 1,
               chess.square_rank(sq), chess.square_file(sq)] = 1.0

    them = not mover
    if board.has_kingside_castling_rights(mover):
        planes[12, :, :] = 1.0
    if board.has_queenside_castling_rights(mover):
        planes[13, :, :] = 1.0
    if board.has_kingside_castling_rights(them):
        planes[14, :, :] = 1.0
    if board.has_queenside_castling_rights(them):
        planes[15, :, :] = 1.0

    if board.ep_square is not None:
        ep = chess.square_mirror(board.ep_square) if flip else board.ep_square
        planes[16, chess.square_rank(ep), chess.square_file(ep)] = 1.0

    planes[17, :, :] = min(board.halfmove_clock, 100) / 100.0
    planes[18, :, :] = 1.0
    return planes
