from enum import Enum
from typing import Optional

import numpy as np

from numba import njit, jit
from scipy.signal.sigtools import _convolve2d
import scipy as sp

from typing import Callable, Tuple, List

BoardPiece = np.int8  # The data type (dtype) of the board
NO_PLAYER = BoardPiece(0)  # board[i, j] == NO_PLAYER where the position is empty
PLAYER1 = BoardPiece(1)  # board[i, j] == PLAYER1 where player 1 has a piece
PLAYER2 = BoardPiece(2)  # board[i, j] == PLAYER2 where player 2 has a piece

board_symbols = {
    NO_PLAYER: '.',
    PLAYER1: 'X',
    PLAYER2: 'O'}

CONNECT_N = 4  # required connected pieces to win

PlayerAction = np.int8  # The column to be played


class GameState(Enum):
    IS_WIN = 1
    IS_DRAW = -1
    STILL_PLAYING = 0

def initialize_game_state(boardShape=(6,7), fillValue=0) -> np.ndarray:
    """
    Returns an ndarray, shape (6, 7) and data type (dtype) BoardPiece, initialized to 0 (NO_PLAYER).
    """
    return np.full(shape=boardShape, fill_value=fillValue, dtype=BoardPiece)

def pretty_print_board(board: np.ndarray) -> str:
    """
    Should return `board` converted to a human readable string representation,
    to be used when playing or printing diagnostics to the console (stdout). The piece in
    board[0, 0] should appear in the lower-left. Here's an example output:
    |==============|
    |              |
    |              |
    |    X X       |
    |    O X X     |
    |  O X O O     |
    |  O O X X     |
    |==============|
    |0 1 2 3 4 5 6 |
    """

    # string of border row, '|==============|'
    border = '|{}|'.format('=' * (2 * (board.shape[1]) - 1))

    # string of column labels row, '|0 1 2 3 4 5 6 |'
    col_label = '|{}|'.format(' '.join([str(n) for n in range(board.shape[1])]))

    # list of strings board rows, e.g. '|  O O X X     |'
    rows = ['|{}|'.format(''.join(str(board[row])[1:-1])) for row in range(board.shape[0])[::-1]]

    #TODO: replace 0,1,2 with symbols for printing

    # for key, value in board_symbols:

    # combine string elements together with \n join
    pretty_rows = '\n'.join(rows)
    pretty_full = '\n'.join([border, pretty_rows, border, col_label])

    return pretty_full

def string_to_board(pp_board: str) -> np.ndarray:
    """
    Takes the output of pretty_print_board and turns it back into an ndarray.
    This is quite useful for debugging, when the agent crashed and you have the last
    board state as a string.
    """

    rows = pp_board.split('\n')
    rows = rows[1:-2] # exclude borders
    rows = rows[::-1] # reverse order
    # assuming the printed board values are the same as the BoardPieces
    board = [list(map(int, r[1:-1].split(' '))) for r in rows]
    return np.asarray(board, dtype=BoardPiece)

def apply_player_action(
    board: np.ndarray, action: PlayerAction, player: BoardPiece, copy: bool = False
) -> np.ndarray:
    """
    Sets board[i, action] = player, where i is the lowest open row. The modified
    board is returned. If copy is True, makes a copy of the board before modifying it.
    """
    # max index of empty column
    column = board[:, action]
    try:
        lowest_open = np.min(np.argwhere(column == 0))
    except ValueError:
        #TODO: signal that an invalid move was attempted
        pass

    # return changed board, or copy of
    if copy:
        out = board.copy()
        out[lowest_open, action] = player
        return out
    else:
        board[lowest_open, action] = player
        return board


col_kernel = np.ones((CONNECT_N, 1), dtype=BoardPiece)
row_kernel = np.ones((1, CONNECT_N), dtype=BoardPiece)
dia_l_kernel = np.diag(np.ones(CONNECT_N, dtype=BoardPiece))
dia_r_kernel = np.array(np.diag(np.ones(CONNECT_N, dtype=BoardPiece))[::-1, :])

# TODO: get njit working for connected_four
# @njit()
def connected_four(
    board: np.ndarray, player: BoardPiece, last_action: Optional[PlayerAction] = None,
) -> bool:
    """
    Returns True if there are four adjacent pieces equal to `player` arranged
    in either a horizontal, vertical, or diagonal line. Returns False otherwise.
    If desired, the last action taken (i.e. last column played) can be provided
    for potential speed optimisation.
    """

    # from Owen Mackwood's connected_four_convolve
    board = board.copy()

    other_player = BoardPiece(player % 2 + 1)
    flat = board.flatten()  # to play nice with numba have to use 1d boolean indexing
    flat[flat == other_player] = BoardPiece(0)
    flat[flat == player] = BoardPiece(1)
    board = flat.reshape(board.shape)

    # board[board == other_player] = BoardPiece(0)
    # board[board == player] = BoardPiece(1)
    # TODO: add condition for if last_action provided
    # if last_action:
    #     result = _convolve2d(board, kernel, 1
    #     , 0, 0, BoardPiece(0))

    for kernel in (col_kernel, row_kernel, dia_l_kernel, dia_r_kernel):
        result = _convolve2d(board, kernel, 1, 0, 0, BoardPiece(0))
        # result = np.convolve(board, kernel)
        if np.any(result == CONNECT_N):
            return True
    return False


def check_end_state(
    board: np.ndarray, player: BoardPiece, last_action: Optional[PlayerAction] = None,
) -> GameState:
    """
    Returns the current game state for the current `player`, i.e. has their last
    action won (GameState.IS_WIN) or drawn (GameState.IS_DRAW) the game,
    or is play still on-going (GameState.STILL_PLAYING)?
    """
    # Check if the player has a connected four
    if connected_four(board=board, player=player, last_action=last_action):
        return GameState(1)
    # Check if no valid moves remain -- that is, if all board locations are occupied by a player
    elif board.all():
        # TODO: replace with get_valid_moves returning that there are none
        return GameState(-1)
    # Otherwise, game is still ongoing
    else:
        return GameState(0)


def get_valid_moves(
    board: np.ndarray
) -> np.ndarray:
    """
    Return the available moves for a given board.

    :param board:
    :param player:
    :param last_action:
    :return:
    """
    top_row = board[-1]
    open_moves = np.argwhere(top_row == NO_PLAYER).squeeze()

    return open_moves


class SavedState:
    pass


GenMove = Callable[
    [np.ndarray, BoardPiece, Optional[SavedState]],  # Arguments for the generate_move function
    Tuple[PlayerAction, Optional[SavedState]]  # Return type of the generate_move function
]