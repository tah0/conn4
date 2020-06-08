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


def minimax_value(board: np.ndarray, player: BoardPiece, maxing: bool, depth: int) -> float:
    """

    :param board:
    :param player:
    :param depth:
    :return:
    """
    other_player = BoardPiece(player % 2 + 1)
    valid_moves = get_valid_moves(board)
    value = 0

    if depth == 0:
        return evaluate_end_state(board, player)
    elif maxing is True:
        value = -np.inf
        for _, move in enumerate(valid_moves):
            # print('Maxing')
            # print('move:', move)
            MMv = minimax_value(board=apply_player_action(board, move, player, copy=True),
                                                player=player,
                                                maxing=False,
                                                depth=depth - 1)
            # print('MM value:', MMv)
            value = max(value, MMv)
    else:
        value = np.inf
        for _, move in enumerate(valid_moves):
            # print('Mining')
            # print('move:', move)
            MMv = minimax_value(board=apply_player_action(board, move, player, copy=True),
                                                player=player,
                                                maxing=True,
                                                depth=depth - 1)
            # print('MM value:', MMv)
            value = min(value, MMv)
    return value


def alpha_beta_value(
        board: np.ndarray,
        player: BoardPiece,
        maxing: bool,
        depth: int,
        alpha,
        beta) -> float:
    other_player = BoardPiece(player % 2 + 1)
    valid_moves = get_valid_moves(board)

    if depth == 0:
        return evaluate_end_state(board, player)
    elif maxing is True:
        value = -np.inf
        for _, move in enumerate(valid_moves):
            ABv = alpha_beta_value(board=apply_player_action(board, move, player, copy=True),
                                player=player,
                                maxing=False,
                                depth=depth - 1,
                                alpha=alpha,
                                beta=beta)
            value = max(value, ABv)
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return value
    else:
        value = np.inf
        for _, move in enumerate(valid_moves):
            ABv = alpha_beta_value(board=apply_player_action(board, move, player, copy=True),
                                player=player,
                                maxing=True,
                                depth=depth - 1,
                                alpha=alpha,
                                beta=beta)
            value = min(value, ABv)
            beta = min(beta, value)
            if beta <= alpha:
                break
        return value


def generate_move_minimax(
    board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState]
) -> Tuple[PlayerAction, Optional[SavedState]]:
    """
    :param board:
    :param player:
    :param saved_state:
    :return:
    """
    open_moves = get_valid_moves(board)
    scores = [minimax_value(apply_player_action(board, move, player, copy=True), player, True, MAX_DEPTH) for move in open_moves]
    scores = [alpha_beta_value(apply_player_action(board, move, player, copy=True), player, True, MAX_DEPTH, alpha=-np.inf, beta=np.inf) for move in open_moves]
    print(scores)
    # TODO: randomly select among best choices
    # best_moves = open_moves[np.argwhere(scores == np.max(scores))]
    # action = np.random.choice(best_moves)
    action = open_moves[np.argmax(scores)]
    # TODO: what to do with saved_state?
    return action, saved_state
