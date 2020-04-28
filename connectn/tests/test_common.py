import numpy as np

def test_initialize_game_state():
    from connectn.agents.common import initialize_game_state

    ret = initialize_game_state()

    assert isinstance(ret, np.ndarray)
    assert ret.dtype == np.int8
    assert ret.shape == (6, 7)
    assert np.all(ret == 0)

def test_pretty_print_board():
    from connectn.agents.common import pretty_print_board
    from connectn.agents.common import initialize_game_state
    board = initialize_game_state()

    ret = pretty_print_board(board)
    assert isinstance(ret, str)


def test_apply_player_action():
    # board: np.ndarray, action: PlayerAction, player: BoardPiece, copy: bool = False
    from connectn.agents.common import apply_player_action
    from connectn.agents.common import initialize_game_state
    board = initialize_game_state()

    ret = apply_player_action(board, action, player, copy)
    assert isinstance(ret, np.ndarray)


def test_string_to_board():
    from connectn.agents.common import string_to_board
    #pp_board: str
    ret = string_to_board(pp_board)
    assert isinstance(ret, np.ndarray)

def test_connected_four():
    pass
