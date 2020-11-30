from collections import defaultdict
import math
import torch.multiprocessing as mp
import os
import copy
from random import choice, randrange
from selfplay import gnu_score
import time
import torch
import random
from bokeNet import ValueNet, value, PolicyNet, policy_dist, features
import go

MAX_TURNS = 70 
EXPAND_THRESH = 10 
EXPAND_NUM =30 

class MCTS:
    "Monte Carlo tree searcher. First rollout the tree then choose a move."

    def __init__(self,
                 value_net: ValueNet=None,
                 policy_net: PolicyNet=None,
                 exploration_weight=1,
                 value_net_weight=0.5):
        self.Q = defaultdict(int)  # total reward of each node
        self.N = defaultdict(int)  # total visit count for each node
        self.V = defaultdict(float)  # accumulated value net evaluations
        self.children = dict()  # children of each node
        self.value_net = value_net
        self.policy_net = policy_net
        self.exploration_weight = exploration_weight
        self.value_net_weight = value_net_weight

    # def __copy__(self):
    #     new_MCTS = MCTS(value_net=self.value_net, policy_net=self.policy_net,
    #                     exploration_weight=self.exploration_weight,
    #                     value_net_weight=self.value_net_weight)
    #     new_MCTS.Q = self.Q.copy()
    #     new_MCTS.N = self.N.copy()
    #     new_MCTS.V = self.V.copy()
    #     new_MCTS.children = self.children.copy()
    #     return new_MCTS
        
    def choose(self, node):
        "Choose the best successor of node. (Choose a move in the game)"
        if node.terminal:
            raise RuntimeError(f"choose called on terminal node {node}")

        if node not in self.children:
            return node.find_random_child(self.policy_net)

        def score(n):
            if self.N[n] == 0:
                return float("-inf")  # avoid unseen moves
            return self.N[n]

        # Choose most visited node
        best = max(self.children[node], key=score)
        return best

    # def do_rollout(self, node, n = 1):
    #     "Train for n iterations"
    #     for _ in range(n):
    #         # Get path to leaf of current search tree
    #         path = self._descend(node)
    #         leaf = path[-1]
    #         if leaf.features is None:
    #             leaf.set_features()
    #         if self.value_net and not leaf.value:
    #             leaf.set_value(self.value_net)
    #         # Get result of rollout starting from leaf
    #         score = self._simulate(leaf, gnu = True)
    #         self._backpropagate(path, score, leaf.value)

    def root_parallel_rollouts(self, root_node, n_workers, n_per):
        with mp.Manager() as manager:
            #using three dicts bc lists have weird behavior
            N = manager.dict({repr(node): visits for node, visits in self.N.items()})
            Q = manager.dict({repr(node): rewards for node, rewards in self.Q.items()})
            V = manager.dict({repr(node): vals for node, vals in self.Q.items()})

            processes = []
            for _ in range(n_workers):
                p = mp.Process(target = do_rollout, 
                                args = (copy.deepcopy(self), copy.deepcopy(root_node), n_per),
                                kwargs = {'V': V, 'N': N, 'Q': Q})
                p.start()
                processes.append(p)
            for p in processes:
                p.join()
            #TODO
            #build combined tree
            #cache policy distributions from different processes
            
    def _descend(self, node):
        "Return a path from root down to leaf via PUCT selection"
        # Start at root (current position)
        path = [node]
        while True:
            # Is node a leaf?
            if node not in self.children or not self.children[node]:
                if self.N[node] > EXPAND_THRESH:
                    self._expand(node)
                return path
            node = self._puct_select(node)  # descend a layer deeper
            path.append(node)

    def _expand(self, node):
        "Update the `children` dict with the children of `node`"
        if node in self.children:
            return  # already expanded
        self.children[node] = node.find_children(self.policy_net)

    # Need to make this faster (ideally at least 10x)
    def _simulate(self, node, gnu = False):
        '''Returns the reward for a random simulation (to completion) of node
        optional: 
            gnu: score with gnugo (default False)''' 
        invert_reward = not node.color
        while True:
            if node.terminal:
                reward = node.reward(gnu)
                reward = invert_reward^reward
                return reward
            node = node.find_random_child(self.policy_net)

    def _backpropagate(self, path, reward, leaf_val = None, kwargs): 
        '''Send the reward back up to the ancestors of the leaf
        optional:
            leaf_val: ValueNet evaluation of leaf
        kwargs:
            (for multiprocessing)
            N: shared dict storing total visits
            Q: shared dict storing total rewards
            V: shared dict storing total value
        '''
        N = kwargs.get(h_N)
        Q = kwargs.get(h_Q)
        V = kwargs.get(h_V)
        
        for node in reversed(path):
            self.N[node] += 1
            self.Q[node] += reward
            if self.value_net:
                self.V[node] += leaf_val

            if h_N != None:
                if repr(node) in h_N:
                    h_N[repr(node)] += 1
                    h_Q[repr(node)] += reward
                    if self.value_net:
                        h_V[repr(node)] += leaf_val
                else:
                    h_N[repr(node)] = 1
                    h_Q[repr(node)] = reward
    
            reward = 1 - reward

    def _puct_select(self, node):
        "Select a child of node with PUCT"
        # Predictor + UCT (PUCT) variant used in AlphaGo
        total_visits = sum(self.N[n] for n in self.children[node])
        # First visit selects policy's top choice
        if total_visits == 0:
            total_visits = 1
        if node.dist is None:
            node.set_dist(self.policy_net)
        def puct(n):
            last_move_prob = node.dist.probs[n.last_move].item()
            if self.value_net is not None:
                avg_reward = 0 if self.N[n] == 0 else ((1 - self.value_net_weight) * self.Q[n]
                                                        + self.value_net_weight * self.V[n]) / self.N[n]
            else:
                avg_reward = 0 if self.N[n] == 0 else self.Q[n]/self.N[n]
            return avg_reward + (self.exploration_weight
                    * last_move_prob 
                    * math.sqrt(total_visits) / (1 + self.N[n]))

        return max(self.children[node], key=puct)

# Now a helper function for parallelized rollouts
def do_rollout(tree, node, n, **kwargs):
    '''Do n rollouts on tree from root node and add tree stats to shared dictionaries'''
    N = kwargs.get(N)
    Q = kwargs.get(Q)
    V = kwargs.get(V)
    for _ in range(n):
        # Get path to leaf of current search tree
        path = tree._descend(node)
        print(f" I AM PROCESS {os.getpid()} FEAR ME")

        leaf = path[-1]
        if leaf.features is None:
            leaf.set_features()
        if tree.value_net and not leaf.value:
            leaf.set_value(tree.value_net)
        # Get result of rollout starting from leaf
        score = tree._simulate(leaf, gnu = True)
        tree._backpropagate(path, score, leaf.value,**kwargs)

class Go_MCTS(go.Game):
    """Wraps go.Game to turn it into a node for search tree expansion
    in MCTS

    Implements all abstract methods from Node as well as a few helper
    functions for determining if the game is legally finished.

    Attributes:
        terminal: a boolean indicating whether the game is legally finished
        color: a boolean indicating the current player's color;
               True = Black, False = White
        dist: torch.distribution.Categorical from policy net
        features: (27,9,9) torch.Tensor, stores the input features of the board state
    """
    def __init__(self, board=go.EMPTY_BOARD, ko=None, turn=0, moves=[],
                 sgf=None, terminal=False,
                 color=True, last_move=None, komi = 5.5, device = "cpu"):
        super().__init__(board, ko, last_move, turn, moves, komi, sgf)
        self.terminal = terminal 
        self.color = color
        self.dist = None
        self.features = None
        self.value = None
        self.device = device 

    def __eq__(self, other):
        return self.board == other.board and self.ko == other.ko

    def __copy__(self):
        return Go_MCTS(board=self.board, ko=self.ko, turn=self.turn,
                       moves=self.moves, terminal=self.terminal,
                       color=self.color, last_move=self.last_move,
                       komi = self.komi, device = self.device)

    def __hash__(self):
        return super().__hash__()

    def __repr__(self):
        return repr( (self.board, self.ko, self.self.last_move

    def find_children(self, policy):
        '''Returns a set of boards (Go_MCTS objects) derived from top policy moves'''
        if self.terminal:
            return set()      
        return {self.make_move(i) for i in self.topk_moves(policy, EXPAND_NUM) if self.is_legal(i)}
    
    def find_random_child(self, policy: PolicyNet):
        '''Draws legal move from distribution given by policy. If no
        policy is given, a legal move is drawn uniformly.
        Returns a copy of the board (Go_MCTS object) after the move has been played.'''
        if self.terminal:
            return self # Game is over; no moves can be made
        return self.make_move(self.get_move(policy)) 

    def topk_moves(self, policy: PolicyNet, k):
        if self.dist is None:
            self.set_dist(policy)
        topk = torch.topk(self.dist.probs, k = k).indices
        return topk.tolist()

    def reward(self, gnu = False):
        '''Returns 1 if Black wins, 0 if White wins.'''
        return gnu_score(self) if gnu else int(self.score() > 0)

    def make_move(self, index):
        '''Returns a copy of the board (Go_MCTS object) after the move
        given by index has been played.'''
        game_copy = copy.copy(self)
        game_copy.play_move(index)
        game_copy.last_move = index
        # It's now the other player's turn
        game_copy.color = not self.color
        game_copy.terminal = game_copy.is_game_over()
        return game_copy

    def get_move(self, policy: PolicyNet):
        '''Sample a move from the policy. If that is illegal or fills player's own eye, find a different
        move from the top policy moves.''' 
        move = self.dist_sample(policy)
        color = go.BLACK if self.color else go.WHITE
        k = 0
        while not self.is_legal(move) or go.possible_eye(self.board, move) == color:
            if k >= 81: return -1
            moves = self.topk_moves(policy, 81)
            move = moves[k]
            k += 1 
        return move

    def is_game_over(self):
        '''Terminate after MAX_TURNS or if last move is PASS''' 
        return self.turn > MAX_TURNS or self.last_move == -1

    def set_dist(self, policy: PolicyNet):
        '''Set the probability distribution for this board'''
        if self.features is None:
            self.set_features()
        self.dist = policy_dist(policy, self, device = self.device, fts=self.features)
    
    def dist_sample(self, policy: PolicyNet):
        '''Sample a move from the policy distribution'''
        if self.dist is None:
            self.set_dist(policy)
        return self.dist.sample().item()

    def set_features(self):
        '''Set the policy features for this board'''
        self.features = features(self)

    def set_value(self, value_net: ValueNet):
        '''Set the value net valuation for this board'''
        self.value = value(value_net, self, device = self.device, fts=self.features)
