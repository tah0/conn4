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
    other_player = BoardPiece(player % 2 + 1)

    end = check_end_state(board, player)
    if end == GameState.IS_WIN:  # win state
        return np.inf
    elif end == GameState.IS_DRAW:  # draw state
        return 0
    # TODO: avoid checking the end state twice
    elif check_end_state(board, other_player) == GameState.IS_WIN:  # lose state
        return -np.inf
    else:  # still playing, use heuristic
        return heuristic(board, player)


def negamax(
        board: np.ndarray,
        player: BoardPiece,
        depth: int,
        ) -> float:
    """

    This is "colorless" negamax -- it assumes the heuristic value is from the perspective of the player its called on
    :param board:
    :param player:
    :param depth:
    :return:
    """
    # if we're at an end state,
    if (depth == 0) or check_game_over(board):
        return evaluate_end_state(board, player)

    # otherwise loop over child nodes
    other_player = BoardPiece(player % 2 + 1)
    value = -np.inf
    for move in get_valid_moves(board):
        value = max(value, -negamax(apply_player_action(board, move, player, copy=True),
                                    other_player,
                                    depth - 1))
    # print(f'value:{value}')
    # print(f'depth = {depth}; end state = {check_game_over(board)}; player = {player}')
    # print(f'move:{move}; max value:{value}')
    return value


def negamax_alpha_beta(
        board: np.ndarray,
        player: BoardPiece,
        depth: int,
        alpha: float,
        beta: float ) -> float:

    # if we're at an end state,
    if (depth == 0) or check_game_over(board):
        return evaluate_end_state(board, player)

    # otherwise loop over child nodes
    other_player = BoardPiece(player % 2 + 1)
    value = -np.inf
    for move in get_valid_moves(board):
        value = max(value, -negamax_alpha_beta(apply_player_action(board, move, player, copy=True),
                                               other_player,
                                               depth - 1,
                                               -beta,
                                               -alpha))
        alpha = max(alpha, value)
        if alpha >= beta:
            break
    # print(f'value:{value}')
    # print(f'depth = {depth}; end state = {check_game_over(board)}; player = {player}')
    # print(f'move:{move}; max value:{value}')
    return value


def generate_move_negamax(
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
        # print(f'playing action {action} for a win')
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
        # print(f'playing action {action} for a block')
        return action, saved_state

    # otherwise, use the heuristic function to score possible states
    scores = [negamax_alpha_beta(state, player, MAX_DEPTH, alpha=-np.inf, beta=np.inf) for state in new_states]

    # randomly select among best moves
    if np.sum(scores == np.max(scores)) > 1:
        best_moves = open_moves[np.argwhere(scores == np.max(scores))].squeeze()
        action = np.random.choice(best_moves)
    else:
        action = open_moves[np.argmax(scores)].squeeze()
    # print(f'Heuristic values: {scores}')
    # print(f'playing action {action} with heuristic value {np.max(scores)}')
    return action, saved_state