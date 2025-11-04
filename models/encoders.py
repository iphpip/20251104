import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List

class TCNEncoder(nn.Module):
    """时序卷积网络编码器"""
    
    def __init__(self, input_dim: int = 1, hidden_dims: list = None,
                 output_dim: int = 128, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [64, 128, 256]
            
        layers = []
        in_channels = input_dim
        
        for i, out_channels in enumerate(hidden_dims):
            dilation = 2 ** i
            padding = (kernel_size - 1) * dilation // 2
            
            layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size, 
                         dilation=dilation, padding=padding),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            in_channels = out_channels
        
        self.conv_layers = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, input_dim)
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        x = x.transpose(1, 2)  # (batch, input_dim, seq_len)
        x = self.conv_layers(x)
        x = self.global_pool(x).squeeze(-1)
        return self.output_layer(x)


class LSTMEncoder(nn.Module):
    """LSTM编码器"""
    
    def __init__(self, input_dim: int = 1, hidden_dim: int = 256, 
                 num_layers: int = 2, output_dim: int = 128, 
                 dropout: float = 0.2, bidirectional: bool = False):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                           batch_first=True, dropout=dropout,
                           bidirectional=bidirectional)
        
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.output_layer = nn.Linear(lstm_output_dim, output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, input_dim)
        if x.dim() == 2:
            x = x.unsqueeze(-1)
            
        lstm_out, (hidden, _) = self.lstm(x)
        
        # 使用最后一个时间步的输出
        if self.bidirectional:
            output = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            output = hidden[-1]
            
        return self.output_layer(output)


class TransformerEncoder(nn.Module):
    """Transformer编码器"""
    
    def __init__(self, input_dim: int = 1, d_model: int = 128, nhead: int = 8,
                 num_layers: int = 2, dim_feedforward: int = 512,
                 output_dim: int = 128, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        
        self.d_model = d_model
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout, max_len)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.output_layer = nn.Linear(d_model, output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, input_dim)
        if x.dim() == 2:
            x = x.unsqueeze(-1)
            
        x = self.input_projection(x)
        x = self.pos_encoding(x)
        
        # Transformer期望的形状: (seq_len, batch, d_model) 或使用batch_first=True
        x = self.transformer(x)  # (batch, seq_len, d_model)
        
        x = x.transpose(1, 2)  # (batch, d_model, seq_len)
        x = self.global_pool(x).squeeze(-1)  # (batch, d_model)
        
        return self.output_layer(x)


class PositionalEncoding(nn.Module):
    """位置编码"""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)