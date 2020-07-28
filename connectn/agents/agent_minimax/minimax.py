from agents.common import BoardPiece, SavedState, PlayerAction, NO_PLAYER, CONNECT_N, GameState
from agents.common import get_valid_moves, apply_player_action, check_end_state, check_game_over
from agents.heuristic import low_row_heuristic, negamax_heuristic
from typing import Optional, Tuple
import numpy as np

MAX_DEPTH = 4

def evaluate_end_state(board: np.ndarray, player: BoardPiece, heuristic=negamax_heuristic) -> float:
    """
    :param heuristic:
    :param board:
    :param player:
    :return:
    """
    end = check_end_state(board, player)
    other_player = BoardPiece(player % 2 + 1)
    if end == GameState.IS_WIN:  # win state
        return np.inf
    elif end == GameState.IS_DRAW:  # draw state
        return 0
    # TODO: workaround to exclude checking the end state twice
    elif check_end_state(board, other_player) == GameState.IS_WIN:
        return -np.inf
    else:
        return heuristic(board, player)


def minimax_value(board: np.ndarray, player: BoardPiece, maxing: bool, depth: int) -> float:
    """

    :param board:
    :param player:
    :param maxing:
    :param depth:
    :return:
    """
    other_player = BoardPiece(player % 2 + 1)
    valid_moves = get_valid_moves(board)
    value = 0

    if depth == 0 or check_game_over(board):
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

    if depth == 0 or check_game_over(board):
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
    print(f'Open moves: {open_moves}')

    new_states = [apply_player_action(board, move, player, copy=True) for move in open_moves]

    # if a move results in a win, play it
    winning_moves = np.array([check_end_state(state, player) for state in new_states]) == GameState.IS_WIN
    if np.any(winning_moves):
        actions = open_moves[np.argwhere(winning_moves)].squeeze()
        if actions.size > 1:
            action = np.random.choice(actions)
        else:
            action = actions
        print(f'playing action {action} for a win')
        return action, saved_state

    # if a move results in blocking an opponent's win, play it
    other_player = BoardPiece(player % 2 + 1)

    new_states_other = [apply_player_action(board, move, other_player, copy=True) for move in open_moves]
    blocking_moves = np.array([check_end_state(state, other_player) for state in new_states_other]) == GameState.IS_WIN
    if np.any(blocking_moves):
        actions = open_moves[np.argwhere(blocking_moves)].squeeze()
        if actions.size > 1:
            action = np.random.choice(actions)
        else:
            action = actions
        print(f'playing action {action} for a block')
        return action, saved_state

    # otherwise, use the heuristic function to score possible states

    # scores = [minimax_value(apply_player_action(board, move, player, copy=True), player, True, MAX_DEPTH) for move in open_moves]
    scores = [alpha_beta_value(apply_player_action(board, move, player, copy=True), player, True, MAX_DEPTH, alpha=-np.inf, beta=np.inf) for move in open_moves]

    # randomly select among best moves
    if np.sum(scores == np.max(scores)) > 1:
        best_moves = open_moves[np.argwhere(scores == np.max(scores))].squeeze()
        action = np.random.choice(best_moves)
    else:
        action = open_moves[np.argmax(scores)].squeeze()
    print(f'Heuristic values: {scores}')
    print(f'playing action {action} with heuristic value {np.max(scores)}')
    return action, saved_state
