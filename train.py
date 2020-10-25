#!/usr/bin/python3
from tqdm import tqdm, trange
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from bokePolicy import PolicyNet, NinebyNineGames

if __name__ == "__main__":
    print("Loading data...")
    data = NinebyNineGames("data", max_games = 20000)
    dataloader = DataLoader(data, batch_size = 128, shuffle = True)
    pi = PolicyNet()
    err = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(pi.parameters(), lr = 0.01)
    print("Size of dataset: {}".format(len(data)))
    
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
            if i % 1000 == 999 :
                print("Epoch: {}, Loss: {loss:.2f}".format(epoch+1, loss = running_loss))
                running_loss = 0.0

