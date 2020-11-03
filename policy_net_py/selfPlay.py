import go
import os
from copy import copy
from pre_process import write_sgf
from bokePolicy import *
import torch
from torch.distributions.categorical import Categorical

def playout(pi_1, pi_2, pos = None):
    '''Play game between policies pi_1 and pi_2, with pi_1 playing black and pi_2 playing white.''' 
    if pos:
        game = copy(pos)
    else:
        game = go.Game()

    while game.turn < 72:
        game.play_move(sample(pi_1, g))
        game.play_move(sample(pi_2, g))

    return game.score() > 0

def sample(policy, game, return_dist = False):
    fts = features(game, policy.scale).unsqueeze(0)
    probs = F.softmax(policy(fts), dim = 1)
    m = Categorical(probs)
    mv = m.sample().item()
    while not game.is_legal(mv):
        mv = m.sample().item()
    if return_dist:
        return mv, m
    return mv 

def update_policy():
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

if __name__ == "__main__":
    pi = PolicyNet()
    device = torch.device("cuda:0" if torch.cuda_is_available() else "cpu")
    pi.to(device)
    checkpt = torch.load("v0.5/policy_v0.5_2020-10-29_1.pt", map_location = device) 
    pi.load_state_dict(checkpt["model_state_dict"])
    pi_opp = copy(pi)
    pi.train()

    optimizer = torch.optim.Adam(pi.parameters(), lr = 0.001)

    games = 100
    for i in range(games):
        g = go.Game()
        for _ in range(35):
            #opponent policy plays black
            g.play_move(sample(pi_opp,g))
            mv, m = sample(pi, g, return_dist = True)
            g.play_move(mv)
            with torch.no_grad():
                res = playout(pi_opp, pi ,g) 
            reward = (-1 if res else 1) #+1 if W won, -1 if W lost

            #Policy gradient descent
            loss = -m.log_prob(mv) * reward 
            loss.backward()
            optimizer.step()
    

    
