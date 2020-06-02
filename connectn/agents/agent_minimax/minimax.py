from agents.common import BoardPiece, SavedState, PlayerAction, NO_PLAYER
from agents.common import get_valid_moves
from typing import Optional, Tuple
import numpy as np


def heuristic(board: np.ndarray, player: BoardPiece) -> float:
    """
    :param board:
    :param player:
    :return:
    """




def minimax_value(board: np.ndarray, player: BoardPiece, depth: int) -> float:
    """

    :param board:
    :param player:
    :param depth:
    :return:
    """

    if depth == 0:
        return heuristic(board, player)
    # elif:
    # else:

# function minimax(node, depth, maximizingPlayer) is
#     if depth = 0 or node is a terminal node then
#         return the heuristic value of node
#     if maximizingPlayer then
#         value := −∞
#         for each child of node do
#             value := max(value, minimax(child, depth − 1, FALSE))
#         return value
#     else (* minimizing player *)
#         value := +∞
#         for each child of node do
#             value := min(value, minimax(child, depth − 1, TRUE))
#         return value


def generate_move_minimax(
    board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState]
) -> Tuple[PlayerAction, Optional[SavedState]]:
    """
    :param board:
    :param player:
    :param saved_state:
    :return:
    """

    top_row = board[-1]
    open_moves = np.argwhere(top_row == NO_PLAYER).squeeze()
    action = np.random.choice(open_moves)






    # TODO: what to do with saved_state?
    return action, saved_state


