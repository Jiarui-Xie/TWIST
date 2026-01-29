import torch
import torch.nn as nn

class MoELayer(nn.Module):
    def __init__(self, 
                 input_dim: int, 
                 output_dim: int, 
                 num_experts: int):
        
        super().__init__()
        self.num_experts = num_experts
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # initialize experts' weights and biases
        self.expert_weights = nn.ParameterList([
            nn.Parameter(torch.randn(input_dim, output_dim) * 0.01)
            for _ in range(num_experts)
        ])
        self.expert_biases = nn.ParameterList([
            nn.Parameter(torch.zeros(output_dim))
            for _ in range(num_experts)
        ])
    
    def forward(self, x, coeffs):
        """
        x: [batch, input_dim]
        coeffs: [batch, num_experts] - from gating network
        output: [batch, output_dim]
        """
        batch_size = x.shape[0]
        
        # expert权重堆叠: [num_experts, input_dim, output_dim]
        all_weights = torch.stack([w for w in self.expert_weights], dim=0)
        all_biases = torch.stack([b for b in self.expert_biases], dim=0)
        
        # coeffs: [batch, num_experts] -> [batch, num_experts, 1, 1]
        coeffs_w = coeffs.unsqueeze(-1).unsqueeze(-1)
        # coeffs: [batch, num_experts] -> [batch, num_experts, 1]
        coeffs_b = coeffs.unsqueeze(-1)
        
        # 加权组合: [batch, input_dim, output_dim]
        combined_weights = (coeffs_w * all_weights.unsqueeze(0)).sum(dim=1)
        # [batch, output_dim]
        combined_biases = (coeffs_b * all_biases.unsqueeze(0)).sum(dim=1)
        
        # 矩阵乘法: [batch, input_dim] @ [batch, input_dim, output_dim] -> [batch, output_dim]
        output = torch.bmm(x.unsqueeze(1), combined_weights).squeeze(1) + combined_biases
        
        return output