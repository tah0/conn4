from agents.common import BoardPiece, SavedState, PlayerAction, PLAYER1, PLAYER2, GameState
from agents.common import get_valid_moves, apply_player_action, check_end_state, check_game_over, evaluate_end_state
from typing import Optional, Tuple, Union
import numpy as np


class MonteCarloNode:
    """
    A Monte Carlo search tree node -- extends the game board with information for MCTS
    """
    def __init__(self, board: np.ndarray, to_play: BoardPiece, last_move: PlayerAction = None, parent=None):
        # board
        self.board = board

        # parent
        self.to_play = to_play  # which player's turn it is
        self.last_move = last_move  # what move resulted in the board
        self.parent = parent  # parent node -- the previous state

        # children methods
        self.children = {}  # dict of children resulting from valid moves
        self.legal_moves = get_valid_moves(board)
        self._unexpanded_moves = list(get_valid_moves(board))  # moves not evaluated yet
        self.expanded_moves = []

        # MCTS methods
        self.n_plays = 0
        self.n_wins = 0
        # self.n_losses = 0

    # experimenting with using @property...
    @property
    def unexpanded_moves(self) -> list:
        """
        Return which moves have not been expanded yet.
        :return: list of unexpanded moves (it's a list so we can pop it, later)
        """
        #     return [m for m in self.legal_moves if m not in self.expanded_moves]
        if self._unexpanded_moves is None:
            self._unexpanded_moves = list(get_valid_moves(self.board))
        else:
            return self._unexpanded_moves

    # def __repr__(self):
    #     return f"{self.__class__.__name__}" # ({self.}, value={self._i})
    #
    # inheriting array methods?
    # def __array__(self):
    #     return self.board

    def get_child(self, move: PlayerAction):
        """
        Return the node associated with making a particular move from the current state.

        :param move: a valid move for this node's state.
        :return: the child node
        """
        return self.children[move]

    def expand(self, move: PlayerAction):
        """
        Expand the child node for a move; creates a new MonteCarloNode associated with the resulting state.

        :param move: a valid move for this node's state.
        :return: the resulting node
        """
        new_board = apply_player_action(board=self.board, action=move, player=self.to_play, copy=True)
        new_node = MonteCarloNode( new_board, to_play=BoardPiece(self.to_play % 2 + 1), last_move=move, parent=self)
        self.children[move] = new_node
        self.expanded_moves.append(move)
        return new_node

    def is_fully_expanded(self):
        """
        Return if all moves have been expanded.

        :return: Bool
        """
        # return if all legal_moves are in expanded_moves
        # return self.legal_moves == self.expanded_moves
        unex = self.unexpanded_moves
        return len(unex) == 0

    def is_leaf(self):
        """
        Return if terminal, not including wins (i.e., if drawn)

        :return: Bool
        """
        return self.board.all()

    def is_game_over(self):
        """
        Check if this node is a terminal game state (including draws, wins).
        :return: Bool
        """
        return check_game_over(self.board)

    def getUCB1(self, bias = 1.4):
        """
        Compute the UCB1 value for the given node.

        :param bias: exploration factor in UCB1
        :return:
        """
        # return ((self.n_wins - self.n_losses) / self.n_plays) + bias * np.sqrt((2 * np.log(self.parent.n_plays) / self.n_plays))
        return (self.n_wins  / self.n_plays) + bias * np.sqrt((2 * np.log(self.parent.n_plays) / self.n_plays))


class MonteCarlo:
    """ Monte Carlo Tree class for holding nodes, for organizing MCTS."""
    def __init__(self, player):
        self.nodes = {}
        self.player = player

    def make_node(self, state: np.ndarray, to_play: BoardPiece):
        """
        Create a node (usually root node) from a given state if it doesn't exist in self.nodes.

        :param state:
        :param to_play: which player's turn it is
        :return:
        """
        if (hash(state.tostring()) + hash(to_play)) not in self.nodes:
            self.nodes[hash(state.tostring()) + hash(to_play)] = MonteCarloNode(state, to_play)

    def run_search(self, state: np.ndarray, to_play: BoardPiece, n_sims=10000):
        """
        Select moves until an unexpanded, non-terminal node is reached,

        :param state:
        :param to_play: which player's turn it is
        :param n_sims:
        :return:
        """
        self.make_node(state, to_play)

        for _ in range(n_sims):
            node = self.select(state, to_play)
            if check_game_over(node.board) is False:
                node = self.expand(node)
                winner = self.simulate(node)
            self.backpropagate(node, winner)

    def select(self, state: np.ndarray, to_play: BoardPiece):
        """
        Select

        :param state:
        :param to_play: which player's turn it is
        :return:
        """
        node = self.nodes[hash(state.tostring()) + hash(to_play)]

        while node.is_fully_expanded() and (not node.is_leaf()):
            scores = [ node.get_child(a).getUCB1() for a in node.legal_moves ]
            best_move = node.legal_moves[np.argmax(scores)]
            node = node.get_child(best_move)
        return node

    def expand(self, node: MonteCarloNode):
        """
        Expand a child of an un-fully expanded node. Randomly selects from unexpanded moves.

        :param node:
        :return:
        """
        i = np.random.randint(len(node.unexpanded_moves))
        play = node.unexpanded_moves.pop(i)
        child_node = node.expand(play)
        self.nodes[hash(child_node.board.tostring()) + hash(child_node.to_play)] = child_node
        return child_node

    def simulate(self, node: MonteCarloNode):
        """
        Simulate a game from a given node -- outcome is either player or GameState.IS_DRAW

        :param node:
        :return:
        """
        current_rollout_state = node.board.copy()
        # print(node.board)
        curr_player = node.to_play
        while not check_game_over(current_rollout_state):
            possible_moves = get_valid_moves(current_rollout_state)
            # print(f'possible: {possible_moves}, {type(possible_moves)}')
            if possible_moves.size > 1:
                action = np.random.choice(list(possible_moves))
            else:
                action = possible_moves
                # print(f'action:{action}')

            # try:
            current_rollout_state = apply_player_action(current_rollout_state, action, curr_player, copy=True)
            # except ValueError:
            #     print('error in applying action')
            #     # print(f'state:{current_rollout_state}')

            curr_player = BoardPiece(curr_player % 2 + 1)
        return evaluate_end_state(current_rollout_state)

    def backpropagate(self, node: MonteCarloNode, winner: Union[BoardPiece, GameState]):
        """
        Backpropagate the outcome of a simulated game through the game tree.

        :param node:
        :param winner:
        :return:
        """
        node.n_plays += 1
        # check if the last player is equal to the winner at this state
        # if BoardPiece(node.to_play % 2 + 1) == winner:
        if winner == self.player:
            node.n_wins += 1
        if node.parent:
            self.backpropagate( node.parent, winner )

    def best_play(self, state: np.ndarray, to_play: BoardPiece):
        """
        Select the best move based on sampling stats.

        Currently: selects the child with highest n_plays ("robust")

        :param state: board
        :param to_play: which player's turn it is
        :return: associated move
        """
        self.make_node(state, to_play)

        if self.nodes[hash(state.tostring()) + hash(to_play)].is_fully_expanded() is False:
            raise ValueError('Trying to best_play without full expansion!')

        node = self.nodes[hash(state.tostring()) + hash(to_play)]

        scores = [node.get_child(a).n_plays for a in node.legal_moves]

        # scores = [self.nodes[]
        #                for a in node.legal_moves]

        best_move = node.legal_moves[np.argmax(scores)]
        return best_move, scores

    def get_stats(self, state, to_play):
        node_hash = hash(state.tostring()) + hash(to_play)
        node = self.nodes[node_hash]
        stats = {'n_plays': node.n_plays,
                 'n_wins' : node.n_wins,
                 'children':[{'play': c.last_move,
                              'n_plays': c.n_plays,
                              'n_wins' : c.n_wins} for c in node.children.values()] }
        return stats

def generate_move_mcts(
    board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState]
) -> Tuple[PlayerAction, Optional[SavedState]]:
    """

    """
    mcts = MonteCarlo(player)
    mcts.run_search(board, player)
    action, scores = mcts.best_play(board, player)
    print(mcts.get_stats(board, player))
    return action, saved_state