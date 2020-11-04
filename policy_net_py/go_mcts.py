import copy
from random import choice, randrange
import time
import torch

from bokePolicy import PolicyNet, policy_dist
import go
from mcts import MCTS, Node

MAX_TURNS = 70

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
                 color=True, last_move=None):
        super().__init__(board, ko, turn, moves, sgf)
        self.policy = policy
        self.terminal = terminal 
        self.color = color
        self.last_move = last_move
        self.dist = None

    def __eq__(self, other):
        return self.board == other.board

    def __hash__(self):
        return hash(self.board + str(self.turn) + str(self.ko)) 

    def __copy__(self):
        return Go_MCTS(board=self.board, ko=self.ko, turn=self.turn,
                       moves=self.moves, policy=self.policy,
                       terminal=self.terminal, color=self.color,
                       last_move=self.last_move)
    
    def find_children(self):
        '''Returns a set of boards (Go_MCTS objects) derived from legal
        moves'''
        if self.terminal:
            return set()      
        return {self.make_move(i) for i in self.get_all_legal_moves()}
    
    def find_random_child(self):
        '''Draws legal move from distribution given by policy. If no
        policy is given, a legal move is drawn uniformly.
        Returns a copy of the board (Go_MCTS object) after the move has
        been played.'''
        if self.terminal:
            return self # Game is over; no moves can be made
        return self.make_move(self.get_move()) 

    def reward(self):
        '''Returns 1 if Black wins, 0 if White wins.'''
        return int(self.score() > 0)

    def make_move(self, index):
        '''Returns a copy of the board (Go_MCTS object) after the move
        given by index has been played.'''
        game_copy = copy.copy(self)
        game_copy.play_move(index)
        game_copy.last_move = index
        # It's now the other player's turn
        game_copy.color = not self.color
        # Check if the move ended the game
        game_copy.terminal = game_copy.is_game_over()
        return game_copy

    def is_terminal(self):
        return self.terminal

    # VERY expensive call, need improve this.
    def get_all_legal_moves(self):
        return [i for i in range(go.N ** 2) if self.is_legal(i)]

    def get_move(self):
        if self.policy: 
            move = self.dist_sample()
            while not self.is_legal(move):
                move = self.dist_sample() 
            return move
        else:
            move = randrange(0, go.N ** 2)
            while not self.is_legal(move):
                move = randrange(0, go.N ** 2)
            return move 

    # Do not use until we figure out how to best terminate the game
    def is_game_over(self):
        '''Terminate after MAX_TURNS or if policy wants to play an illegal move'''
        return self.turn > MAX_TURNS

    def set_dist(self):
        '''Set the probability distribution for this board'''
        self.dist = policy_dist(self.policy, self)
    
    def dist_sample(self):
        '''Sample a move from the policy distribution'''
        if not self.dist:
            self.set_dist()
        return self.dist.sample().item()

if __name__ == '__main__':
    pi = PolicyNet()
    checkpt = torch.load("v0.5/RL_policy_3.pt", map_location = torch.device("cpu"))
    pi.load_state_dict(checkpt["model_state_dict"])
    NUMBER_OF_ROLLOUTS = 100
    tree = MCTS(exploration_weight = 1)
    board = Go_MCTS(policy=pi)
    print(board)
    while True:
        while True:
            try:
                row_col = input("enter move: ")
                if row_col == 'q':
                    break
                sq_c = 9*(ord(row_col[0])-65) + int(row_col[1]) - 1
                if board.is_legal(sq_c):
                    break
            except:
                print("Enter a valid option, or type 'q' to quit")
        if row_col == 'q':
            break
        board = board.make_move(sq_c)
        print(board)
        if board.terminal:
            break
        tree.do_rollout(board, NUMBER_OF_ROLLOUTS)
        board = tree.choose(board)

        print(board)
        if board.terminal:
            break
