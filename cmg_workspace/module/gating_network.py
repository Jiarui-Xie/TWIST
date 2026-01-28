import torch
import torch.nn as nn

class GatingNetwork(nn.Module):
    def __init__(self, 
                 input_dim: int, 
                 num_experts: int, 
                 hidden_dim: int):
        super().__init__()

        self.gating_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, num_experts),
            nn.Softmax(dim=-1) # w1, w2 ... adds up to 1 
        )
    
    def forward(self, x):
        # x: [batch, input_dim]
        # output: [batch, num_experts]
        return self.gating_network(x)