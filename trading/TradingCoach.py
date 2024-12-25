# trading/TradingCoach.py
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from TradingEnv import TradingEnv
from CustomPPOPolicy import CustomPPOPolicy
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--batch_size', type=int, default=64, help='批次大小')
    parser.add_argument('--total_timesteps', type=int, default=100000, help='总训练步数')
    parser.add_argument('--embed_dim', type=int, default=64, help='嵌入维度')
    parser.add_argument('--num_heads', type=int, default=4, help='注意力头数')
    parser.add_argument('--ff_dim', type=int, default=128, help='前馈网络维度')
    parser.add_argument('--num_layers', type=int, default=2, help='Transformer层数')
    parser.add_argument('--window_size', type=int, default=50, help='时间窗口大小')
    args = parser.parse_args()

    # 加载预处理后的数据
    sequences = np.load('scaled_train_data.npy')  # [num_sequences, window_size, features]
    
    # 划分训练和测试集
    split = int(0.8 * len(sequences))
    train_data = sequences[:split]
    test_data = sequences[split:]
    
     # 创建自定义环境
    class TradingEnvWrapper(TradingEnv):
        def __init__(self, data, window_size=50):
            super(TradingEnvWrapper, self).__init__(data, window_size)
    window_size = 50
    env = TradingEnvWrapper(sequences, window_size=window_size)
    env = DummyVecEnv([lambda: env])

    # 初始化 PPO 模型，使用自定义策略
    model = PPO(CustomPPOPolicy, env, verbose=1, learning_rate=args.lr, batch_size=args.batch_size)

    # 训练模型
    model.learn(total_timesteps=args.total_timesteps)

    # 保存模型
    model.save("ppo_trading_model")

    print("训练完成并保存模型为 'ppo_trading_model'")
    
    

if __name__ == "__main__":
    main()
