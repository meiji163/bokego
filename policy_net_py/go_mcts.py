from copy import copy
from random import choice

from bokePolicy import PolicyNet, policy_predict
import go
from mcts import MCTS, Node


class Go_MCTS(go.Game, Node):
    """Wraps go.Game to turn it into a Node for search tree expansion
    in MCTS

    Implements all abstract methods from Node as well as a few helper
    functions for determining if the game is legally finished.

    Attributes:
        policy: a PolicyNet for move selection
        terminal: a boolean indicating whether the game is legally finished
        color: a boolean indicating the current player's color;
               True = Black, False = White
    """
    def __init__(self, board=go.EMPTY_BOARD, ko=None, turn=0, moves=[],
                 sgf=None, policy: PolicyNet=None, terminal=False,
                 color=True):
        super().__init__(board, ko, turn, moves, sgf)
        self.policy = policy
        self.terminal = terminal
        self.color = color

    def __eq__(node1, node2):
        return node1.board == node2.board

    def __hash__(self):
        return self.board.__hash__()

    def __copy__(self):
        return Go_MCTS(board=self.board, ko=self.ko, turn=self.turn,
                       moves=self.moves, policy=self.policy,
                       terminal=self.terminal, color=self.color)
    
    def find_children(self):
        '''Returns a set of boards (Go_MCTS objects) derived from legal
        moves'''
        if self.terminal:
            return set()      
        return {self.make_move(i) for i in self.get_legal_moves()}
    
    def find_random_child(self):
        '''Draws legal move from distribution given by policy. If no
        policy is given, a legal move is drawn uniformly.
        Returns a copy of the board (Go_MCTS object) after the move has
        been played.'''
        if self.terminal:
            return self # Game is over; no moves can be made
        if self.policy:
            move = policy_predict(self.policy, self) # NEEDS TO BE FIXED
        else:
            move = choice(self.get_legal_moves())  
        return self.make_move(move)

    def reward(self):
        '''Returns 1 for a win, 0 for a loss.'''
        if not self.terminal:
            raise RuntimeError(f"reward called on nonterminal board {self}")
        # Black = True, White = False
        return int(self.color and self.current_winner())

    def make_move(self, index):
        '''Returns a copy of the board (Go_MCTS object) after the move
        given by index has been played.'''
        game_copy = copy(self)
        game_copy.play_move(index)
        # It's now the other player's turn
        game_copy.color = not self.color
        # Check if the move ended the game
        game_copy.terminal = game_copy.is_game_over()
        return game_copy

    def is_terminal(self):
        return self.terminal

    # VERY expensive call, need improve this.
    def get_legal_moves(self):
        return [i for i in range(go.N ** 2) if self.is_legal(i)]

    def is_game_over(self):
        '''Game is over if there are no more legal moves
        (or if both players pass consecutively, or if a
        player resigns...)'''
        return len(self.get_legal_moves()) == 0

    def current_winner(self):
        return self.score() > 0

if __name__ == '__main__':
    NUMBER_OF_ROLLOUTS = 50
    tree = MCTS()
    board = Go_MCTS()
    print(board)
    while True:
        while True:
            try:
                row_col = input("enter 'row col': ")
                index = go.squash(tuple([int(i) for i in row_col.split(' ') ]))
                if board.is_legal(index):
                    break
            except:
                print("Enter a valid option")
        board = board.make_move(index)
        print(board)
        if board.terminal:
            break
        for _ in range(NUMBER_OF_ROLLOUTS):
            tree.do_rollout(board)
        board = tree.choose(board)
        print(board)
        if board.terminal:
            break