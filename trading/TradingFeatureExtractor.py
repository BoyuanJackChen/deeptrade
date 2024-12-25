# trading/TradingFeatureExtractor.py
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
import torch

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # [max_len, 1, embed_dim]
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

class TransformerFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, embed_dim=64, num_heads=4, ff_dim=128, num_layers=2, window_size=50, feature_dim=5):
        super(TransformerFeatureExtractor, self).__init__(observation_space, features_dim=embed_dim)
        self.embed = nn.Linear(feature_dim, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, max_len=window_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)  # 将序列维度池化为1

    def forward(self, observations):
        # observations: [batch, window_size, feature_dim]
        x = self.embed(observations)  # [batch, window_size, embed_dim]
        x = x.permute(1, 0, 2)  # [window_size, batch, embed_dim]
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)  # [window_size, batch, embed_dim]
        x = x.permute(1, 2, 0)  # [batch, embed_dim, window_size]
        x = self.pool(x).squeeze(-1)  # [batch, embed_dim]
        return x
