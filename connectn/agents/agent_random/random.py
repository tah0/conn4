from agents.common import BoardPiece, SavedState, PlayerAction, NO_PLAYER
from agents.common import get_valid_moves
from typing import Optional, Tuple
import numpy as np

def generate_move_random(
    board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState]
) -> Tuple[PlayerAction, Optional[SavedState]]:
    """
    Choose a valid, non-full column randomly and return it as `action`
    :param board:
    :param player:
    :param saved_state:
    :return:
    """
    open_moves = get_valid_moves(board)
    action = np.random.choice(open_moves)
    # TODO: what to do with saved_state?
    return action, saved_state