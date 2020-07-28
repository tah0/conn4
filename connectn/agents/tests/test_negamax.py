import numpy as np
from agents.common import BoardPiece, NO_PLAYER, PLAYER1, PLAYER2, PlayerAction, GameState


def test_evaluate_end_state():
    from agents.agent_negamax.negamax import evaluate_end_state
    from agents.common import initialize_game_state

    dummy_board = initialize_game_state()
    assert evaluate_end_state(dummy_board, PLAYER1) == 0
    assert evaluate_end_state(dummy_board, PLAYER2) == 0

    win_player_1 = initialize_game_state()
    win_player_1[:4, 0] = PLAYER1
    assert evaluate_end_state(win_player_1, PLAYER1) == np.inf
    assert evaluate_end_state(win_player_1, PLAYER2) == -np.inf


def test_negamax():
    from agents.agent_negamax.negamax import negamax
    from agents.common import initialize_game_state

    dummy_board = initialize_game_state()
    assert negamax(dummy_board, PLAYER1, 0) == 0
    assert negamax(dummy_board, PLAYER2, 0) == 0

    win_player_1 = initialize_game_state()
    win_player_1[:4, 0] = PLAYER1
    assert negamax(win_player_1, PLAYER1, 0) == np.inf
    assert negamax(win_player_1, PLAYER2, 0) == -np.inf

    near_1_win = initialize_game_state()
    near_1_win[:3, 0] = PLAYER1
    assert negamax(near_1_win, PLAYER1, 1) == np.inf


def test_low_row_heuristic():
    from agents.agent_negamax.negamax import low_row_heuristic
    from agents.common import initialize_game_state

    dummy_board = initialize_game_state()
    assert low_row_heuristic(dummy_board, PLAYER1) == 0
    assert low_row_heuristic(dummy_board, PLAYER2) == 0

    dummy_board[0,0] = PLAYER1
    assert low_row_heuristic(dummy_board, PLAYER1) == 5
    assert low_row_heuristic(dummy_board, PLAYER2) == 0

    dummy_board[0,1] = PLAYER2
    assert low_row_heuristic(dummy_board, PLAYER1) == 5
    assert low_row_heuristic(dummy_board, PLAYER2) == 5

    dummy_board[0,2] = PLAYER2
    assert low_row_heuristic(dummy_board, PLAYER1) == 5
    assert low_row_heuristic(dummy_board, PLAYER2) == 10

    dummy_board[0,3] = PLAYER1
    assert low_row_heuristic(dummy_board, PLAYER1) == 10
    assert low_row_heuristic(dummy_board, PLAYER2) == 10

    dummy_board[1,0] = PLAYER1
    assert low_row_heuristic(dummy_board, PLAYER1) == 14
    assert low_row_heuristic(dummy_board, PLAYER2) == 10

    dummy_board[1,1] = PLAYER2
    assert low_row_heuristic(dummy_board, PLAYER1) == 14
    assert low_row_heuristic(dummy_board, PLAYER2) == 14