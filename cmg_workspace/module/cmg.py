import torch
import torch.nn as nn
from .gating_network import GatingNetwork
from .moe_layer import MoELayer

class CMG(nn.Module):
    """Conditional Motion Generator with Mixture-of-Experts"""
    
    def __init__(
        self,
        motion_dim: int,
        command_dim: int,
        hidden_dim: int,
        num_experts: int,
        num_layers: int,
    ):
        super().__init__()
        
        self.motion_dim = motion_dim
        self.command_dim = command_dim
        input_dim = motion_dim + command_dim
        
        # Gating network
        self.gating = GatingNetwork(input_dim, num_experts, hidden_dim)
        
        # MoE layers (3层MLP)
        self.layers = nn.ModuleList()
        
        # Layer 1: input -> hidden
        self.layers.append(MoELayer(input_dim, hidden_dim, num_experts))
        
        # Layer 2: hidden -> hidden
        for _ in range(num_layers - 2):
            self.layers.append(MoELayer(hidden_dim, hidden_dim, num_experts))
        
        # Layer 3: hidden -> output
        self.layers.append(MoELayer(hidden_dim, motion_dim, num_experts))
        
        self.activation = nn.ELU()
    
    def forward(self, motion: torch.Tensor, command: torch.Tensor) -> torch.Tensor:
        """
        motion: [batch, motion_dim] - 当前帧的关节位置+速度
        command: [batch, command_dim] - 速度指令
        return: [batch, motion_dim] - 预测的下一帧
        """
        # 拼接输入
        x = torch.cat([motion, command], dim=-1)
        
        # Gating计算系数（整个前向传播用同一组系数）
        coeffs = self.gating(x)
        
        # 通过MoE layers
        for i, layer in enumerate(self.layers):
            x = layer(x, coeffs)
            # 最后一层不加激活
            if i < len(self.layers) - 1:
                x = self.activation(x)
        
        return x
    