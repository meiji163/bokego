from collections import defaultdict
from math import sqrt
import numpy as np
import torch
import torch.multiprocessing as mp
from torch.distributions import dirichlet, categorical 
import os
from copy import copy, deepcopy
import bokego.nnet as nnet
from bokego.nnet import ValueNet, PolicyNet, policy_dist
import bokego.go as go

MAX_TURNS = 80 

class MCTS:
    '''Monte Carlo tree searcher. 
    Nodes are selected with the PUCT variant used in AlphaGo.
    First rollout the tree then choose. To input a move
    use make_move(..) and set_root(..)

    args:
        root (Go_MCTS): node representing current game state
        policy_net (PolicyNet): PolicyNet for getting prior distributions
        value_net (ValueNet): for getting board value (between -1 and 1)
                   If no valuenet is given, rewards are based on simulations only
        no_sim (bool): disable simulations and evaluate only with value net 
    kwargs:
        expand_thresh (int): number of visits before leaf is expanded (default 100)
        branch_num (int): number of children to expand. If not specified, all legal moves expanded 
        exploration_weight (float): scalar for prior prediction (default 4.0)
        value_net_weight (float): scalar between 0 and 1 for mixing value network
                                  and simulation rewards (default 0.5)
        noise_weight (float): scalar between 0 and 1 for adding Dirichlet noise (default 0.25)
    Attributes:
        Q: dict containing total simulation rewards of each node
        N: dict containing total visits to each node
        V: dict containing accumulated value of each node
        children: dict containing children of each node
    '''
   
    _dirichlet = dirichlet.Dirichlet(0.1*torch.ones(go.N**2))
    _val_cache = dict()
    _dist_cache = dict()
    _fts_cache = dict()

    def __init__(self, root, 
                policy_net: PolicyNet=None, 
                value_net: ValueNet=None, 
                **kwargs):
        self.Q = defaultdict(int)  
        self.N = defaultdict(int)  
        self.V = defaultdict(float) 
        self.children = dict()  
        if policy_net is None:
            raise TypeError("Missing required keywork argument: 'policy_net'")
        self.policy_net = policy_net
        self.value_net = value_net
        self.no_sim = kwargs.get("no_sim", False)
        if self.value_net is None and self.no_sim:
            raise TypeError("Keyword argument 'value_net' is required for no simulation mode")
        self.expand_thresh = kwargs.get("expand_thresh",100)
        self.branch_num = kwargs.get("branch_num")
        self.exploration_weight = kwargs.get("exploration_weight", 4.0)
        self.noise_weight = kwargs.get("noise_weight", 0)
        if self.no_sim:
            self.value_net_weight = 1.0
        elif self.value_net is None:
            self.value_net_weight = 0.0
        else:
            self.value_net_weight = kwargs.get("value_net_weight",  0.5)
        
        #for GPU computations
        self.device = kwargs.get("device", torch.device("cpu"))
        policy_net.to(self.device)
        if value_net != None:
            value_net.to(self.device)

        #initialize the root
        self.set_root(root)

    def __deepcopy__(self, memo):
        cls = self.__class__
        new_tree = cls.__new__(cls)
        new_tree.__dict__.update(self.__dict__)
        new_tree.root = deepcopy(self.root)
        new_tree.V = deepcopy(self.V)
        new_tree.Q = deepcopy(self.Q)
        new_tree.N = deepcopy(self.N)
        new_tree.children = deepcopy(self.children) 
        return new_tree

    #For pickling
    def __getstate__(self):
        state_dict = self.__dict__.copy()
        del state_dict["policy_net"] 
        del state_dict["value_net"] 
        return state_dict
    
    def __setstate__(self, state_dict):
        self.__dict__.update(state_dict)
        #give the nodes a reference to the tree
        for n in self.children:
            n.tree = self    
            for c in self.children[n]:
                c.tree = self
        #set the policy net and value net manually
        self.policy_net = None  
        self.value_net = None 
        
    def choose(self, node = None):
        '''Choose the best child of root and set it as the new root
        optional:
            node: choose from different node (doesn't affect root)''' 
        if node is None:
            node = self.root
        if node._terminal:
            #print(f"{node} Board is terminal")
            return node
        if node not in self.children:
            return node.find_random_child()

        def score(n):
            if self.N[n] == 0:
                return float("-inf")  # avoid unseen moves
            return self.N[n]

        # Choose most visited node
        best = max(self.children[node], key=score)
        if node == self.root:
            self.set_root(best)
        return best

    def rollout(self, n = 1, analyze_dict = None):
        '''Do rollouts from the root
        args:
            n (int): number of rollouts
            analyze_dict: (optional) dict to store variations
        '''
        for _ in range(n):
            # Get path to leaf of current search tree 
            path = self._descend()
            leaf = path[-1]

            if analyze_dict != None and len(path) > 2:
                analyze_dict[ path[1] ] = path[1:]

            if not self.no_sim:
                score = self._simulate(leaf, gnu = True)
            else:
                score = None 
            self._backpropagate(path, score, leaf.value)

    def set_root(self, node):
        self.root = node 
        self.root.tree = self
        self.root._add_noise(self.noise_weight)
        self._expand(self.root)

    def winrate(self, node = None):
        '''Returns float between 0.0 and 1.0 representing winrate
        from persepctive of the root
        optional:
            node: return winrate of a different node'''
        w = self.value_net_weight
        if node is None:
            node = self.root
        if self.N[node] > 0:
            v = ((1-w)*self.Q[node] + w* self.V[node])/self.N[node]
            return (v+1)/2
        return 0

    def _descend(self):
        "Return a path from root down to leaf via PUCT selection"
        path = [self.root]
        node = self.root
        while True:
            # Is node a leaf?
            if node not in self.children or not self.children[node]:
                if self.N[node] > self.expand_thresh:
                    self._expand(node)
                return path
            node = self._puct_select(node)  # descend a layer deeper
            path.append(node)

    def _expand(self, node):
        "Update the `children` dict with the children of `node`"
        if node in self.children:
            return  # already expanded
        if self.branch_num:
            self.children[node] = node.find_children(k = self.branch_num)
        else:
            self.children[node] = node.find_children()

    # Need to make this faster (ideally at least 10x)
    def _simulate(self, node, gnu = False):
        '''Returns the reward for a random simulation (to completion) of node
        optional: 
            gnu: if True, score with gnugo (default False)''' 
        invert_reward = not (node.turn %2 == 0) #invert if it is white's turn
        while True:
            if node._terminal:
                reward = node.reward(gnu)
                if invert_reward:
                    reward = -reward
                return reward
            node = node.find_random_child()

    def _backpropagate(self, path, reward, leaf_val): 
        '''Send the reward back up to the ancestors of the leaf'''
        for node in reversed(path):
            self.N[node] += 1
            if reward:
                self.Q[node] += reward
                reward = -reward 
            if self.value_net != None:
                self.V[node] += leaf_val

    def _puct_select(self, node):
        "Select a child of node with PUCT"
        total_visits = sum(self.N[n] for n in self.children[node])
        # First visit selects policy's top choice 
        if total_visits == 0:
            total_visits = 1
        def puct(n):
            last_move_prob = node.dist.probs[n.last_move].item()
            avg_reward = 0 if self.N[n] == 0 else \
                    ((1 - self.value_net_weight) * self.Q[n]
                     + self.value_net_weight * self.V[n]) / self.N[n]
            return avg_reward + (self.exploration_weight
                    * last_move_prob 
                    * sqrt(total_visits) / (1 + self.N[n]))

        return max(self.children[node], key=puct)

    def _prune(self):
        '''Prune the tree leaving only root and its descendants'''
        new_children = defaultdict(int)
        q = [self.root]
        while q:
            n = q.pop()
            c = self.children.get(n)
            if c:
                new_children[n] = c 
                q.extend(c)
        self.children = new_children
        remove_me = set()
        for n in self.N:
            if n not in new_children:
                remove_me.add(n)
        for n in remove_me:
            del self.Q[n]
            del self.N[n]
            if n in self.V:
                del self.V[n]

class Go_MCTS(go.Game):
    """Wraps go.Game to turn it into a node for search tree expansion
    in MCTS. The node acesses policy/value net from its tree.

    Implements all abstract methods from Node as well as a few helper
    functions for determining if the game is legally finished.

    args:
        board: the board string
        ko: current ko index
        last_move: last move index
        turn: the turn number (starting from 0)

    Attributes:
        dist: the policy net's distribution for the game state
        value: the value net valuation of the game state
        features: the input features for the game state
    """
    def __init__(self, board=go.EMPTY_BOARD, 
                 ko=None, turn=0, last_move=None):
        super(Go_MCTS,self).__init__(board, ko, last_move, turn)
        self._terminal = self.is_game_over() 
        self.tree = None
    
    def __getstate__(self):
        state_dict = {"board": self.board, "last_move": self.last_move,
                         "ko": self.ko, "turn": self.turn, 
                         "_libs": self._libs, "_hash": self._hash}  
        return state_dict

    def __setstate__(self, state_dict):
        _hash = state_dict.pop("_hash")
        _libs = state_dict.pop("_libs")
        self.__init__(**state_dict)
        self._hash = _hash
        self._libs = _libs
        
    def __eq__(self, other):
        return self.board == other.board and self.ko == other.ko \
                            and self.last_move == other.last_move
    
    def __hash__(self):
        return super(Go_MCTS,self).__hash__()

    def  __deepcopy__(self, memo):
        cls = self.__class__
        new_node = cls.__new__(cls)
        new_node.__dict__.update(self.__dict__)
        for k, v in self.__getstate__().items():
            new_node.__dict__[k] = deepcopy(v)
        return new_node 
    
    def find_children(self, k = None):
        '''Find all children of node.
        optional:
            k: find k children with the top prior probability according to policy'''
        if self._terminal:
            return set()      
        if k != None and 0 <= k < go.N**2:
            return {self.make_move(i) for i in self.topk_moves(k) if self.is_legal(i)}
        return {self.make_move(i) for i in self.get_legal_moves()}
    
    def find_random_child(self):
        '''Draws legal move from distribution given by policy. 
        Returns the board (Go_MCTS object) after the move has been played.'''
        if self._terminal:
           return self # Game is over; no moves can be made
        return self.make_move(self.get_move()) 

    def topk_moves(self, k):
        topk = torch.topk(self.dist.probs, k = k).indices
        return topk.tolist()

    def reward(self, gnu = False):
        '''Returns 1 if Black wins, -1 if White wins.
        optional:
            gnu: score with gnugo'''
        if gnu:
            reward = go.gnu_score(self)
            if reward != None:
                return reward
        return 1 if self.score() > 0 else -1 

    def make_move(self, index):
        '''Returns a copy of the board (Go_MCTS object) after the move
        given by index has been played.'''
        game_copy = deepcopy(self)
        game_copy.play_move(index)
        game_copy._terminal = game_copy.is_game_over()
        return game_copy

    def get_move(self):
        '''Return a move sampled from the policy distribution.
        Pass as a last resort.'''
        move = self.dist.sample().item()
        color = go.BLACK if self.turn%2 == 0 else go.WHITE
        tries = 0
        while not self.is_legal(move) or go.possible_eye(self.board, move) == color:
            if tries >= go.N**2: 
                return go.PASS 
            self.dist.probs[move] = 0 #zero out absurd moves
            move = self.dist.sample().item()  
            tries += 1
        return move

    def is_game_over(self):
        '''Terminate after MAX_TURNS or if last move is pass''' 
        return self.turn > MAX_TURNS or self.last_move == go.PASS 

    def _add_noise(self, weight):
        '''Add Dirichlet noise to the distribution'''
        noise = MCTS._dirichlet.sample()
        self.dist.probs = (1 - weight)*self.dist.probs + weight*noise

    @property
    def dist(self):
        if self.tree is None:
            return
        dist = MCTS._dist_cache.get(self)
        if dist is None:
            dist = policy_dist(self.tree.policy_net, 
                                self, 
                                fts = self.features,
                                device = self.tree.device)
            dist.probs = dist.probs.to(torch.device("cpu"))
            MCTS._dist_cache[self] = dist 
        return dist

    @property
    def features(self):
        fts = MCTS._fts_cache.get(self)
        if fts is None:
            fts = nnet.features(self)
            MCTS._fts_cache[self] = fts 
        return fts 

    @property
    def value(self):
        if self.tree is None or self.tree.value_net is None:
            return
        val = MCTS._val_cache.get(self)
        if val is None:
            val = nnet.value(self.tree.value_net, self, 
                            fts=self.features,
                            device = self.tree.device)
            MCTS._val_cache[self] = val
        return val
    
    @property
    def winrate(self):
        if self.tree is None:
            return
        return self.tree.winrate(self) 
           
