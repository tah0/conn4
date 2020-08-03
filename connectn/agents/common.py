from enum import Enum
from typing import Optional

import numpy as np

from numba import njit, jit
from scipy.signal.sigtools import _convolve2d
import scipy as sp

from typing import Callable, Tuple, List

CONNECT_N = 4  # required connected pieces to win
BoardPiece = np.int8  # The data type (dtype) of the board
NO_PLAYER = BoardPiece(0)  # board[i, j] == NO_PLAYER where the position is empty
PLAYER1 = BoardPiece(1)  # board[i, j] == PLAYER1 where player 1 has a piece
PLAYER2 = BoardPiece(2)  # board[i, j] == PLAYER2 where player 2 has a piece
PlayerAction = np.int8  # The column to be played


class GameState(Enum):
    IS_WIN = 1
    IS_DRAW = -1
    STILL_PLAYING = 0


def initialize_game_state(board_shape=(6, 7), fill_value: BoardPiece = NO_PLAYER) -> np.ndarray:
    """
    Returns an ndarray, shape (6, 7) and data type (dtype) BoardPiece, initialized to fill_value (default NO_PLAYER).
    """
    return np.full(shape=board_shape, fill_value=fill_value, dtype=BoardPiece)


def pretty_print_board(board: np.ndarray, board_symbols: dict = {NO_PLAYER: '.', PLAYER1: 'X', PLAYER2: 'O'}) -> str:
    """
    Returns a readable string of the input board, substituting symbols for player pieces, for example:

    |==============|
    |. . . . . . . |
    |. . . . . . . |
    |. . X X . . . |
    |. . O X X . . |
    |. O X O O . . |
    |. O O X X . . |
    |==============|
    |0 1 2 3 4 5 6 |
    """

    # string of border row, '|==============|'
    border = '|{}|'.format('=' * (2 * (board.shape[1]) - 1))

    # string of column labels row, '|0 1 2 3 4 5 6 |'
    col_label = '|{}|'.format(' '.join([str(n) for n in range(board.shape[1])]))

    # list of strings board rows, e.g. '|  O O X X     |'
    rows = ['|{}|'.format(''.join(str(board[row])[1:-1])) for row in range(board.shape[0])[::-1]]

    # combine string elements together with \n join
    pretty_rows = '\n'.join(rows)
    for piece in (PLAYER1, PLAYER2, NO_PLAYER):
        pretty_rows = pretty_rows.replace(str(piece), board_symbols[piece])
    pretty_full = '\n'.join([border, pretty_rows, border, col_label])

    return pretty_full


def string_to_board(pp_board: str) -> np.ndarray:
    """
    Takes the output of pretty_print_board and turns it back into an ndarray.
    This is quite useful for debugging, when the agent crashed and you have the last
    board state as a string.
    """
    pp_board = pp_board.replace('.', '0')
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
    column = board[:, action].squeeze()
    lowest_open = min(np.argwhere(column == 0))
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

    # other_player = BoardPiece(player % 2 + 1)
    # flat = board.flatten()  # to play nice with numba have to use 1d boolean indexing
    # flat[flat == other_player] = BoardPiece(0)
    # flat[flat == player] = BoardPiece(1)
    # board = flat.reshape(board.shape)

    other_player = BoardPiece(player % 2 + 1)
    board[board == other_player] = NO_PLAYER
    board[board == player] = BoardPiece(1)

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


@njit()
def connected_four_iter(
    board: np.ndarray, player: BoardPiece, last_action: Optional[PlayerAction] = None
) -> bool:
    rows, cols = board.shape
    rows_edge = rows - CONNECT_N + 1
    cols_edge = cols - CONNECT_N + 1
    for i in range(rows):
        for j in range(cols_edge):
            if np.all(board[i, j:j+CONNECT_N] == player):
                return True
    for i in range(rows_edge):
        for j in range(cols):
            if np.all(board[i:i+CONNECT_N, j] == player):
                return True
    for i in range(rows_edge):
        for j in range(cols_edge):
            block = board[i:i+CONNECT_N, j:j+CONNECT_N]
            if np.all(np.diag(block) == player):
                return True
            if np.all(np.diag(block[::-1, :]) == player):
                return True
    return False


CONNECTION = connected_four_iter


# njit makes this _slower_
def check_end_state(
    board: np.ndarray, player: BoardPiece, last_action: Optional[PlayerAction] = None, connected_function = CONNECTION
) -> GameState:
    """
    Returns the current game state for the current `player`, i.e. has their last
    action won (GameState.IS_WIN) or drawn (GameState.IS_DRAW) the game,
    or is play still on-going (GameState.STILL_PLAYING)?
    """
    # Check if the player has a connected four
    # if connected_four(board=board, player=player, last_action=last_action):
    if connected_function(board=board, player=player, last_action=last_action):
        return GameState(1)
        # return GameState.IS_WIN

    # Check if no valid moves remain -- that is, if all board locations are occupied by a player
    elif get_valid_moves(board).size == 0:
        return GameState(-1)
        # return GameState.IS_DRAW
    # Otherwise, game is still ongoing
    else:
        return GameState(0)
        # return GameState.STILL_PLAYING


# njit makes this _slower_
def check_game_over(
    board: np.ndarray, last_action: Optional[PlayerAction] = None, connected_function = CONNECTION
) -> bool:
    """
    Return if the board is a terminal state (either player has won, or a draw).
    :param connected_function:
    :param board:
    :param last_action:
    :return:
    """
    if connected_function(board=board, player=PLAYER1, last_action=last_action):
        return True
    elif connected_function(board=board, player=PLAYER2, last_action=last_action):
        return True
    elif board.all():
        return True
    else:
        return False


# relies on njit-slower func.
def evaluate_end_state(board: np.ndarray):
    """
    Takes a terminal state and returns the winning player (PLAYER1, PLAYER2) or a draw (IS_DRAW)

    :param board: the (game over) state to evaluate
    :return: player who won, or GameState.IS_DRAW

    """
    end = check_end_state(board, PLAYER1)
    if end == GameState.IS_WIN:  # win state
        return PLAYER1
    elif end == GameState.IS_DRAW:  # draw state
        return GameState.IS_DRAW
    elif check_end_state(board, PLAYER2) == GameState.IS_WIN:  # lose state
        return PLAYER2
    else:
        raise ValueError('Evaluating end state on a non-end state!')

@njit()
def get_valid_moves(
    board: np.ndarray
) -> np.ndarray:
    """
    Return the available moves for a given board.

    :param board:
    :return:
    """
    top_row = board[-1] # get open slots in the top row

    # njit-friendly indices
    open_moves = np.nonzero(top_row == NO_PLAYER)[0]

    # old alternative not compatible with njit:
    # open_moves = np.argwhere(top_row == NO_PLAYER).squeeze()
    return open_moves


# def randomly_choose_move(moves: np.ndarray) -> int:
#     """
#     Pick uniformly from among array values if there are more than one. Extends np.random.choice to single value arrays.
#     :param moves:
#     :return:
#     """
#     if moves.size > 1:
#         return np.random.choice(moves)
#     else:
#         return int(moves[0])

class SavedState:
    pass


GenMove = Callable[
    [np.ndarray, BoardPiece, Optional[SavedState]],  # Arguments for the generate_move function
    Tuple[PlayerAction, Optional[SavedState]]  # Return type of the generate_move function
]
