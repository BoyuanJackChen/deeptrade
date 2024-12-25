# trading/evaluate.py
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from TradingEnv import TradingEnv
from CustomPPOPolicy import CustomPPOPolicy
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def evaluate_model(model_path, test_data, window_size=50):
    # 加载模型
    model = PPO.load(model_path)
    
    # 创建自定义环境
    env = TradingEnv(test_data, window_size=window_size)
    env = DummyVecEnv([lambda: env])
    
    # 初始化环境
    obs = env.reset()
    done = False
    total_profit = 0.0  # 初始化为浮点数
    
    # 获取初始环境状态
    initial_balance = env.envs[0].initial_balance
    print(f"Initial Balance: {initial_balance:.2f}")
    
    # 获取初始净资产
    net_worth = initial_balance
    print(f"Step\tAction\t\tBalance\t\tPosition\tNet Worth\tProfit")
    
    step = 0  # 记录当前步骤
    
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        
        # 检查 reward 是否为数组，并提取标量值
        if isinstance(reward, np.ndarray):
            reward = reward[0]  # 提取数组中的第一个元素
        total_profit += reward  # 累加奖励
        
        # 获取当前环境状态
        current_balance = env.envs[0].balance
        current_position = env.envs[0].position
        current_net_worth = env.envs[0].net_worth
        current_profit = env.envs[0].total_profit
        
        # 映射动作索引到动作名称
        action_mapping = {0: "Hold", 1: "Buy", 2: "Sell"}
        action_name = action_mapping.get(action[0], "Unknown")
        
        # 打印当前步骤的信息
        print(f"{step}\t{action_name}\t\t{current_balance:.2f}\t\t{current_position:.4f}\t\t{current_net_worth:.2f}\t\t{current_profit:.2f}")
        
        step += 1  # 增加步骤计数
    
    print(f"\nTotal Profit: {total_profit:.2f}")

if __name__ == "__main__":
    # 加载预处理后的数据
    data = pd.read_csv('trading_data.csv', parse_dates=['date'])
    data = data.sort_values('date').reset_index(drop=True)
    features = ['open', 'high', 'low', 'close', 'volume']
    data_features = data[features].values

    # 归一化
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data_features)

    window_size = 50

    # 划分训练和测试集
    split = int(0.8 * len(scaled_data))
    test_data = scaled_data[split:]

    # 评估模型
    evaluate_model("ppo_trading_model", test_data, window_size=window_size)
