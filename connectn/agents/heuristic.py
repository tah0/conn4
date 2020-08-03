from agents.common import BoardPiece, SavedState, PlayerAction, NO_PLAYER, CONNECT_N, GameState
from agents.common import get_valid_moves, apply_player_action, check_end_state, check_game_over
from typing import Optional, Tuple
import numpy as np

from scipy.signal.sigtools import _convolve2d


# kernel for up n-in-a-row -- to use for open (n-1)-in-a-row
kernels = {}
for n in range(2, CONNECT_N+1):
    col_kernel = np.ones((n, 1), dtype=BoardPiece)
    row_kernel = np.ones((1, n), dtype=BoardPiece)
    dia_l_kernel = np.diag(np.ones(n, dtype=BoardPiece))
    dia_r_kernel = np.array(np.diag(np.ones(n, dtype=BoardPiece))[::-1, :])

    kernels[n] = (col_kernel, row_kernel, dia_l_kernel, dia_r_kernel)

# the score contributions of an open n-in-a-row
weights = {
    1: 1,
    2: 15,
    3: 100,
}


def negamax_heuristic(board: np.ndarray, player: BoardPiece) -> float:
    """
    A heuristic for negamax -- the weighted sum of n-in-a-row for the current board.

    :param board: current board
    :param player: the player to play
    :return: selected move
    """
    board = board.copy()

    other_player = BoardPiece(player % 2 + 1)
    board[board == other_player] = -1
    board[board == player] = 1
    score = 0

    # if a move results in blocking a loss, return it

    for n in range(2, CONNECT_N+1):
        weight = weights[n-1]
        for _, kernel in enumerate(kernels[n]):
            result = _convolve2d(board, kernel, 1, 0, 0, BoardPiece(0))
            score += weight * np.sum(result == (n - 1))
    return score


def low_row_heuristic(board:np.ndarray, player:BoardPiece) -> float:
    """
    A dumb heuristic to play the move with lowest open row.

    :param board: current board
    :param player: the player to play
    :return: selected move
    """
    board = board.copy()
    xx, yy = np.meshgrid(np.arange(board.shape[0]), np.arange(board.shape[1]))
    # xx.T is row
    other_player = BoardPiece(player % 2 + 1)
    board[board == other_player] = 0
    board[board == player] = 1
    weights = xx.T[::-1, :]
    return float(np.sum(board*weights))
