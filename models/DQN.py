import torch
import torch.nn as nn

class DQNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, structure = [64]):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Linear(input_dim, structure[0]),
            nn.ReLU()            
            )
        if len(structure) >= 2:
            for i in range(len(structure)-1):
                self.model.append(nn.Linear(structure[i], structure[i+1]))
                self.model.append(nn.ReLU())
        self.model.append(nn.Linear(structure[-1], output_dim))
        
    def forward(self, x) -> list:  
        out = self.model(x)
        return out