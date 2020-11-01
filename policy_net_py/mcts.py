"""
A minimal implementation of Monte Carlo tree search (MCTS) in Python 3
Kyle Chan, 2020. Copyright Lobachevsky Inc.
"""
from abc import ABC, abstractmethod
from collections import defaultdict
import math
import multiprocessing
from multiprocessing.pool import Pool

EXPAND_THRESH = 10

class MCTS:
    "Monte Carlo tree searcher. First rollout the tree then choose a move."

    def __init__(self, exploration_weight=1):
        self.Q = defaultdict(int)  # total reward of each node
        self.N = defaultdict(int)  # total visit count for each node
        self.children = dict()  # children of each node
        self.exploration_weight = exploration_weight

    def choose(self, node):
        "Choose the best successor of node. (Choose a move in the game)"
        if node.is_terminal():
            raise RuntimeError(f"choose called on terminal node {node}")

        if node not in self.children:
            return node.find_random_child()

        def score(n):
            if self.N[n] == 0:
                return float("-inf")  # avoid unseen moves
            return self.Q[n] / self.N[n]  # average reward

        return max(self.children[node], key=score)

    def do_rollout(self, node, n):
        "Train for n iterations"
        n_workers = multiprocessing.cpu_count()
        outgoing = []  # positions waiting for a playout
        incoming = []  # positions that finished evaluation
        ongoing = []  # currently ongoing playout jobs
        i = 0
        with Pool(processes=n_workers) as pool:
            while i < n:
                if not outgoing:
                    path = self._select(node)
                    outgoing.append(path)

                if len(ongoing) >= n_workers:
                    # Too many playouts running? Wait a bit...
                    ongoing[0][0].wait(0.01 / n_workers)
                else:
                    i += 1
                    leaf = path[-1]
                    # Heuristic: only expand leaf if it's promising (i.e. visited a lot).
                    if self.N[leaf] > EXPAND_THRESH:
                        self._expand(leaf)
                    # Issue a self._simulate job to the worker pool
                    path = outgoing.pop()
                    ongoing.append((pool.apply_async(self._simulate, (leaf,)), path))

                # Anything to store in the tree?  (We do this step out-of-order
                # picking up data from the previous round so that we don't stall
                # ready workers while we update the tree.)
                while incoming:
                    score, path = incoming.pop()
                    self._backpropagate(path, score)

                # Any playouts are finished yet?
                for job, path in ongoing:
                    if not job.ready():
                        continue
                    # Yes! Queue them up for storing in the tree.
                    score = job.get()
                    incoming.append((score, path))
                    ongoing.remove((job, path))


    def _select(self, node):
        "Find an unexplored descendent of `node`"
        path = []
        while True:
            path.append(node)
            if node not in self.children or not self.children[node]:
                # node is either unexplored or terminal
                return path
            unexplored = self.children[node] - self.children.keys()
            if unexplored:
                n = unexplored.pop()
                path.append(n)
                return path
            node = self._uct_select(node)  # descend a layer deeper

    def _expand(self, node):
        "Update the `children` dict with the children of `node`"
        if node in self.children:
            return  # already expanded
        self.children[node] = node.find_children()

    def _simulate(self, node):
        "Returns the reward for a random simulation (to completion) of `node`"
        invert_reward = True 
        while True:
            if node.is_terminal():
                reward = node.reward()
                reward = invert_reward^reward 
                # print(node)
                # print(reward)
                # print(node.score())
                return reward
            node = node.find_random_child()
            invert_reward = not invert_reward

    def _backpropagate(self, path, reward):
        "Send the reward back up to the ancestors of the leaf"
        for node in reversed(path):
            self.N[node] += 1
            self.Q[node] += reward
            reward = 1 - reward  # 1 for me is 0 for my enemy, and vice versa

    def _uct_select(self, node):
        "Select a child of node, balancing exploration & exploitation"

        # All children of node should already be expanded:
        assert all(n in self.children for n in self.children[node])

        log_N_vertex = math.log(self.N[node])

        def uct(n):
            "Upper confidence bound for trees"
            return self.Q[n] / self.N[n] + self.exploration_weight * math.sqrt(
                log_N_vertex / self.N[n]
            )

        return max(self.children[node], key=uct)


class Node(ABC):
    """
    A representation of a single board state.
    MCTS works by constructing a tree of these Nodes.
    Could be e.g. a chess or checkers board state.
    """

    @abstractmethod
    def find_children(self):
        "All possible successors of this board state"
        return set()

    @abstractmethod
    def find_random_child(self):
        "Random successor of this board state (for more efficient simulation)"
        return None

    @abstractmethod
    def is_terminal(self):
        "Returns True if the node has no children"
        return True

    @abstractmethod
    def reward(self):
        "Assumes `self` is terminal node. 1=win, 0=loss, .5=tie, etc"
        return 0

    @abstractmethod
    def __hash__(self):
        "Nodes must be hashable"
        return 123456789

    @abstractmethod
    def __eq__(node1, node2):
        "Nodes must be comparable"
        return True
