from agents.common import BoardPiece, SavedState, PlayerAction, NO_PLAYER, CONNECT_N, GameState
from agents.common import get_valid_moves, apply_player_action, check_end_state
from typing import Optional, Tuple
import numpy as np

from scipy.signal.sigtools import _convolve2d

col_kernel = np.ones((CONNECT_N, 1), dtype=BoardPiece)
row_kernel = np.ones((1, CONNECT_N), dtype=BoardPiece)
dia_l_kernel = np.diag(np.ones(CONNECT_N, dtype=BoardPiece))
dia_r_kernel = np.array(np.diag(np.ones(CONNECT_N, dtype=BoardPiece))[::-1, :])

MAX_DEPTH = 4


def heuristic(board: np.ndarray, player: BoardPiece) -> float:
    """
    :param board:
    :param player:
    :return:
    """
    board = board.copy()

    other_player = BoardPiece(player % 2 + 1)
    board[board == other_player] = -1
    board[board == player] = 1
    score = 0
    for kernel in (col_kernel, row_kernel, dia_l_kernel, dia_r_kernel):
        result = _convolve2d(board, kernel, 1, 0, 0, BoardPiece(0))
    score += np.sum(result == (CONNECT_N - 1))
    return score


def evaluate_end_state(board: np.ndarray, player: BoardPiece) -> float:
    """
    :param board:
    :param player:
    :return:
    """
    other_player = BoardPiece(player % 2 + 1)
    end = check_end_state(board, player)
    if end == GameState.STILL_PLAYING:
        return heuristic(board, player)
    elif end == GameState.IS_WIN:  # win state
        return np.inf
    elif check_end_state(board, other_player) == GameState.IS_WIN:
        return -np.inf
    elif end == GameState.IS_DRAW:  # draw state
        return 0


def negamax(
        board: np.ndarray,
        player: BoardPiece,
        maxing: bool,
        depth: int,
        color: int
        ) -> float:
    """
    :param board:
    :param player:
    :param maxing:
    :param depth:
    :param color:
    :return:
    """
    # if depth == 0:
    #     evaluate_end_state(board)
# function negamax(node, depth, color) is
#     if depth = 0 or node is a terminal node then
#         return color × the heuristic value of node
#     value := −∞
#     for each child of node do
#         value := max(value, −negamax(child, depth − 1, −color))
#     return value
    pass
def negamax_alpha_beta(
        board: np.ndarray,
        player: BoardPiece,
        maxing: bool,
        depth: int,
        color: int,
        alpha,
        beta) -> float:
    pass
# function negamax(node, depth, α, β, color) is
#     if depth = 0 or node is a terminal node then
#         return color × the heuristic value of node
#
#     childNodes := generateMoves(node)
#     childNodes := orderMoves(childNodes)
#     value := −∞
#     foreach child in childNodes do
#         value := max(value, −negamax(child, depth − 1, −β, −α, −color))
#         α := max(α, value)
#         if α ≥ β then
#             break (* cut-off *)
#     return value

def generate_move_negamax():
    pass