# trading/TradingNNet.py
from NeuralNet import NeuralNet
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # [max_len, 1, embed_dim]
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

class TransformerNet(nn.Module):
    def __init__(self, input_dim, embed_dim=64, num_heads=4, ff_dim=128, num_layers=2, window_size=50):
        super(TransformerNet, self).__init__()
        self.embed = nn.Linear(input_dim, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, max_len=window_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)  # 将序列维度池化为1

    def forward(self, x, mask=None):
        x = self.embed(x)  # [batch, seq, embed_dim]
        x = x.transpose(0, 1)  # [seq, batch, embed_dim]
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x, mask=mask)  # [seq, batch, embed_dim]
        x = x.transpose(1, 2)  # [batch, embed_dim, seq]
        x = self.pool(x).squeeze(-1)  # [batch, embed_dim]
        return x

class TradingNeuralNet(NeuralNet):
    def __init__(self, game, args):
        super(TradingNeuralNet, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        input_dim = game.get_init_board().shape[1]  # 特征数量
        self.model = TransformerNet(input_dim, 
                                    embed_dim=args.embed_dim, 
                                    num_heads=args.num_heads, 
                                    ff_dim=args.ff_dim, 
                                    num_layers=args.num_layers, 
                                    window_size=game.env.window_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        self.loss_fn = nn.MSELoss()

    def train_step(self, examples):
        self.model.train()
        for epoch in range(self.args.epochs):
            np.random.shuffle(examples)
            batches = [examples[i:i + self.args.batch_size] for i in range(0, len(examples), self.args.batch_size)]
            for batch in batches:
                states, mcts_probs, rewards = zip(*batch)
                states = torch.tensor(states, dtype=torch.float32).to(self.device)  # [batch, window, features]
                # mcts_probs 在SB3中不需要，因此可以忽略
                rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)  # [batch]

                # 前向传播，仅提取特征
                features = self.model(states)  # [batch, embed_dim]

                # 这里不需要自行计算策略和价值的损失，因为SB3会处理这些
                # 因此，你可以考虑移除 `train_step` 方法，使用SB3的训练流程

    def predict(self, state):
        self.model.eval()
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)  # [1, window, features]
            features = self.model(state)
            return features.cpu().numpy()[0]
