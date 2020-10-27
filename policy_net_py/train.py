#!/usr/bin/python3
from tqdm import tqdm, trange
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from bokePolicy import PolicyNet, NinebyNineGames
import argparse

if __name__ == "__main__":    
    print("Loading data...")
    data = NinebyNineGames("/home/jupyter/BokeGo/data/boards.csv", scale = 100)
    dataloader = DataLoader(data, batch_size = 128, shuffle = True)
    pi = PolicyNet(scale = 100)
    err = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(pi.parameters(), lr = 0.001)
    print("Number of board positions: {}".format(len(data)))
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    pi.to(device)
    
    epochs = 1 
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in tqdm(enumerate(dataloader,0)):
            inputs, moves = data
            inputs, moves = inputs.to(device), moves.to(device)
            
            optimizer.zero_grad()
            outputs = pi(inputs)
            
            #backprop
            loss = err(outputs, moves)
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()
            if i%2000 == 1999:
                print("Loss: ", running_loss)
                running_loss = 0.0
        torch.save({"model_state_dict": pi.state_dict(), "optimizer_state_dict": optimizer.state_dict(), 'epoch': epoch}, r"/home/jupyter/BokeGo/policy_train.pt")
    
