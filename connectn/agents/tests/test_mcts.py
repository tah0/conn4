import numpy as np
from agents.common import BoardPiece, NO_PLAYER, PLAYER1, PLAYER2, PlayerAction, GameState
from agents.common import initialize_game_state, apply_player_action, get_valid_moves
from agents.agent_mcts.mcts import MonteCarloNode, MonteCarlo
import copy
from pytest import raises

""" MonteCarloNode class method tests """

# an initial (blank) board and node
initial_state = initialize_game_state()
initial_node = MonteCarloNode(initial_state, PLAYER1)


# test init, prior to creating further nodes
def test_init_node():
    assert np.all(initial_node.legal_moves == get_valid_moves(initial_state))
    assert np.all(initial_node.legal_moves == initial_node.unexpanded_moves)
    assert np.all(initial_node.board == initial_state)


# manually create a fully expanded node from the initial node
# just relies on the init method (and apply_player_action)
fully_expanded_node = copy.deepcopy(initial_node)

for move in fully_expanded_node.legal_moves:
    new_board = apply_player_action(board=fully_expanded_node.board, action=move, player=fully_expanded_node.to_play, copy=True)
    new_node = MonteCarloNode(new_board, to_play=BoardPiece(fully_expanded_node.to_play % 2 + 1), last_move=move, parent=fully_expanded_node)
    fully_expanded_node.children[move] = new_node

fully_expanded_node.expanded_moves = fully_expanded_node.legal_moves
fully_expanded_node._unexpanded_moves = []


def test_unexpanded_moves():
    assert np.all(initial_node.unexpanded_moves == initial_node.legal_moves)
    assert fully_expanded_node.unexpanded_moves == []


def test_get_child():
    for move in fully_expanded_node.expanded_moves:
        child = fully_expanded_node.get_child(move)
        assert child == fully_expanded_node.children[move]

    # test illegal move -- fail
    move = fully_expanded_node.board.shape[1] + 1
    assert move not in fully_expanded_node.legal_moves
    with raises(KeyError):
        fully_expanded_node.get_child(move)

    # test unexpanded child -- fail
    for move in initial_node.unexpanded_moves:
        with raises(KeyError):
            initial_node.get_child(move)


def test_expand_node():
    node = copy.deepcopy(initial_node)
    for move in node.unexpanded_moves:
        child = node.expand(move)
        assert isinstance(child, MonteCarloNode)
        assert move == child.last_move
        assert child.parent == node
        assert child.to_play == BoardPiece(node.to_play % 2 + 1)

        # test that attributes are all equal for manually-expanded and method-expanded children
        same_child = fully_expanded_node.get_child(move)
        d1 = vars(child)
        d2 = vars(same_child)
        for attribute, value in d1.items():
            if attribute in ['parent']:
                continue
            assert np.all(d2[attribute] == value)


def test_is_fully_expanded():
    node = copy.deepcopy(initial_node)
    for move in node.legal_moves:
        assert not node.is_fully_expanded()
        node.expand(move)
    assert node.is_fully_expanded()


def test_get_ucb1():
    move = fully_expanded_node.expanded_moves[0]
    node = copy.deepcopy(fully_expanded_node).get_child(move)
    node.parent.n_plays = 10
    node.n_wins = 2
    node.n_plays = 5
    bias = 1.4
    assert node.get_ucb1() == (node.n_wins / node.n_plays) + bias * np.sqrt((2 * np.log(node.parent.n_plays) / node.n_plays))


""" MonteCarlo (tree) class method tests """

from agents.agent_mcts.mcts import MonteCarlo
player = PLAYER1


def test_make_node():
    tree = MonteCarlo(player)
    tree.make_node(initial_state, player)
    key = hash(initial_state.tostring()) + hash(player)
    assert isinstance(tree.nodes[key], MonteCarloNode)
    node1 = MonteCarloNode(initial_state, player)
    node2 = tree.nodes[key]
    for attribute, value in vars(node1).items():
        assert np.all(vars(node2)[attribute] == value)


def test_expand():
    tree = MonteCarlo(player)
    tree.make_node(initial_state, player)
    key = hash(initial_state.tostring()) + hash(player)
    root = tree.nodes[key]

    for _ in root.unexpanded_moves:
        child = tree.expand(root)
        assert isinstance(child, MonteCarloNode)
        assert child.last_move in root.legal_moves
        assert child.last_move in root.expanded_moves
        assert child.parent == root
        assert child.to_play == BoardPiece(player % 2 + 1)
        child_key = hash(child.board.tostring()) + hash(child.to_play)
        assert tree.nodes[child_key] == child


def test_simulate():
    tree = MonteCarlo(player)
    tree.make_node(initial_state, player)
    key = hash(initial_state.tostring()) + hash(player)
    root = tree.nodes[key]
    for _ in range(1000):
        outcome = tree.simulate(root)
        assert isinstance(outcome, BoardPiece) or isinstance(outcome, GameState)


def test_select():
    tree = MonteCarlo(player)
    tree.make_node(initial_state, player)


def test_backpropagate():
    tree = MonteCarlo(player)
    tree.make_node(initial_state, player)
    key = hash(initial_state.tostring()) + hash(player)
    root = tree.nodes[key]

    n_test_sims = 100

    for m in root.legal_moves:
        wins = 0
        child = tree.expand(root)
        for i in range(n_test_sims):
            outcome = tree.simulate(child)
            tree.backpropagate(child, outcome)
            if outcome == player:
                wins+=1
        assert child.n_wins == wins
        assert child.n_plays == n_test_sims

    assert root.n_plays == sum([c.n_plays for c in root.children.values()])
    assert root.n_wins == sum([c.n_wins for c in root.children.values()])


def test_run_search():
    tree = MonteCarlo(player)
    tree.make_node(initial_state, player)
    key = hash(initial_state.tostring()) + hash(player)
    root = tree.nodes[key]

    n_sims = 10

    for i in range(n_sims):
        tree.run_search(root.board, root.to_play, 1)

        assert root.n_plays == i+1


def test_best_play():

    # check that best plays are max n_plays
    tree = MonteCarlo(player)
    tree.make_node(initial_state, player)
    key = hash(initial_state.tostring()) + hash(player)
    root = tree.nodes[key]
    tree.run_search(root.board, root.to_play, n_sims=5000)
    # check that best move is the max of n_plays of children
    scores = [root.get_child(a).n_plays for a in root.legal_moves]
    assert tree.best_play(root.board, root.to_play)[0] == root.legal_moves[np.argmax(scores)]

    # check that winning moves are selected
    for c in get_valid_moves(initial_state):
        near_win = copy.deepcopy(initial_state)
        near_win[:3, c] = player
        # print(near_win)
        tree = MonteCarlo(player)
        tree.make_node(near_win, player)
        tree.run_search(near_win, player, n_sims=1000)
        # print(tree.get_stats(near_win, player))
        assert tree.best_play(near_win, player)[0] == c


# generate move function
def test_generate_move_mcts():
    from agents.agent_mcts.mcts import generate_move_mcts
    move, ss = generate_move_mcts(initial_state, player, None)
    assert isinstance(move, np.int64)