import numpy as np
from agents.common import BoardPiece, NO_PLAYER, PLAYER1, PLAYER2, PlayerAction, GameState


def test_initialize_game_state():
    from agents.common import initialize_game_state

    ret = initialize_game_state()

    assert isinstance(ret, np.ndarray)
    assert ret.dtype == BoardPiece
    assert ret.shape == (6, 7)
    assert np.all(ret == NO_PLAYER)


def test_pretty_print_board():
    from agents.common import pretty_print_board
    from agents.common import initialize_game_state

    dummy_board = initialize_game_state()
    ret = pretty_print_board(dummy_board)

    # Split string at newlines
    rows = ret.split('\n')

    # 1. Evaluate board area

    # number of rows (excluding borders)  = 1st dim of board
    assert len(rows[1:-2]) == dummy_board.shape[0]

    # length of each row (excluding border rows) = 2nd dim of board
    assert all((len(r)-1) / 2 == dummy_board.shape[1] for r in rows[1:-2]) # div by 2 to account for space

    # 2. Evaluate border area
    borderForm = '|{}|'.format('=' * (2 * (dummy_board.shape[1]) - 1))
    column_label_form = '|{}|'.format(' '.join([str(n) for n in range(dummy_board.shape[1])]))

    assert rows[0] == borderForm
    assert rows[-2] == borderForm
    assert rows[-1] == column_label_form

    #TODO: test that X and O match up to players


def test_string_to_board():
    from agents.common import string_to_board
    from agents.common import initialize_game_state, pretty_print_board

    dummy_board = initialize_game_state()

    # pretty print the board to get a string
    dummy_string = pretty_print_board(dummy_board)

    # convert string back to board
    ret = string_to_board(dummy_string)

    # new board should match original board
    assert (ret==dummy_board).all()


def test_apply_player_action():
    from agents.common import apply_player_action
    from agents.common import initialize_game_state

    dummy_board = initialize_game_state()
    dummy_board[0,0] = PLAYER1

    test_board = initialize_game_state()
    #with copying
    copied_test_board = apply_player_action(test_board, PlayerAction(0), PLAYER1, copy=True)
    #without copying
    apply_player_action(test_board, PlayerAction(0), PLAYER1)

    assert (copied_test_board == dummy_board).all()
    assert (test_board == dummy_board).all()


def test_connected_four():
    from agents.common import connected_four
    from agents.common import initialize_game_state

    dummy_board = initialize_game_state()

    # check empty board
    assert connected_four(dummy_board, PLAYER1) is False

    # check a horizontal win
    horizontal_win_player1 = dummy_board.copy()
    horizontal_win_player1[0, 0:4] = PLAYER1
    assert connected_four(horizontal_win_player1, PLAYER1) is True

    # check a vertical win
    vertical_win_player1 = dummy_board.copy()
    vertical_win_player1[0:4, 0] = PLAYER1
    assert connected_four(vertical_win_player1, PLAYER1) is True

    # check a diagonal win
    diagonal_win_player1 = dummy_board.copy()
    for i in range(4):
        diagonal_win_player1[i,i] = PLAYER1
    assert connected_four(diagonal_win_player1, PLAYER1) is True

    #TODO: check all possible win states
    #TODO: check for both/all players -- enumerate player
    #TODO: check for false positives -- 4 disconnected,


def test_connected_four_iter():
    from agents.common import connected_four_iter as connected_four
    from agents.common import initialize_game_state

    dummy_board = initialize_game_state()

    # check empty board
    assert connected_four(dummy_board, PLAYER1) is False

    # check a horizontal win
    horizontal_win_player1 = dummy_board.copy()
    horizontal_win_player1[0, 0:4] = PLAYER1
    assert connected_four(horizontal_win_player1, PLAYER1) is True

    # check a vertical win
    vertical_win_player1 = dummy_board.copy()
    vertical_win_player1[0:4, 0] = PLAYER1
    assert connected_four(vertical_win_player1, PLAYER1) is True

    # check a diagonal win
    diagonal_win_player1 = dummy_board.copy()
    for i in range(4):
        diagonal_win_player1[i,i] = PLAYER1
    assert connected_four(diagonal_win_player1, PLAYER1) is True

    #TODO: check all possible win states
    #TODO: check for both/all players -- enumerate player
    #TODO: check for false positives -- 4 disconnected,


def test_check_end_state():
    from agents.common import check_end_state
    from agents.common import initialize_game_state

    dummy_board = initialize_game_state()

    # an empty board
    assert check_end_state(dummy_board, PLAYER1) == GameState(0)

    # a horizontal win
    horizontal_win_player1 = dummy_board.copy()
    horizontal_win_player1[0, 0:4] = PLAYER1
    assert check_end_state(horizontal_win_player1, PLAYER1) == GameState(1)

    # a full board with a win
    full1 = np.full_like(dummy_board, PLAYER1)
    assert check_end_state(full1, PLAYER1) == GameState(1)

    # a full board with no win (other player wins) -- returns Draw
    full2 = np.full_like(dummy_board, PLAYER2)
    assert check_end_state(full2, PLAYER1) == GameState(-1)

    # TODO: a full board with no win for either player
    # TODO: check providing last_action argument behavior


def test_get_valid_moves():
    from agents.common import get_valid_moves
    from agents.common import initialize_game_state

    dummy_board = initialize_game_state()
    all_moves = np.arange(dummy_board.shape[1])

    assert np.all(get_valid_moves(dummy_board) == all_moves)


def test_check_game_over():
    from agents.common import check_game_over
    from agents.common import initialize_game_state

    dummy_board = initialize_game_state()
    player = PLAYER1
    assert check_game_over(dummy_board) is False

    horizontal_win_player1 = dummy_board.copy()
    horizontal_win_player1[0, 0:4] = PLAYER1
    assert check_game_over(horizontal_win_player1) is True

    # check a vertical win
    vertical_win_player1 = dummy_board.copy()
    vertical_win_player1[0:4, 0] = PLAYER1
    assert check_game_over(vertical_win_player1) is True

    # check a diagonal win
    diagonal_win_player1 = dummy_board.copy()
    for i in range(4):
        diagonal_win_player1[i,i] = PLAYER1
    assert check_game_over(diagonal_win_player1) is True

    # TODO: check wins/losses for both players; without a player