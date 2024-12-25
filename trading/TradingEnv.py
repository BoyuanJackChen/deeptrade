
# trading/TradingEnv.py
import gym
from gym import spaces
import numpy as np

class TradingEnv(gym.Env):
    """
    自定义的交易环境，遵循OpenAI Gym接口。
    """

    def __init__(self, data, window_size=50, initial_balance=10000):
        super(TradingEnv, self).__init__()
        self.data = data
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.reset()

        # 动作空间：0=持有, 1=买入, 2=卖出
        self.action_space = spaces.Discrete(3)

        # 观察空间：窗口大小 * 特征数量
        self.observation_space = spaces.Box(low=-1.0, high=1.0, 
                                            shape=(window_size, data.shape[1]),
                                            dtype=np.float32)

    def reset(self):
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.position = 0  # 持仓数量
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        self.total_profit = 0
        return self._get_observation()

    def _get_observation(self):
        return self.data[self.current_step - self.window_size:self.current_step]

    def step(self, action):
        done = False
        reward = 0
        current_price = self.data[self.current_step -1][3]  # 收盘价

        # 执行动作
        if action == 1:  # 买入
            if self.balance > 0:
                self.position += self.balance / current_price
                self.balance = 0
        elif action == 2:  # 卖出
            if self.position > 0:
                self.balance += self.position * current_price
                self.position = 0

        self.current_step += 1

        if self.current_step >= len(self.data):
            done = True

        # 计算当前净资产
        self.net_worth = self.balance + self.position * self.data[self.current_step -1][3]
        self.max_net_worth = max(self.max_net_worth, self.net_worth)
        profit = self.net_worth - self.initial_balance
        reward = profit

        # 环境状态
        next_state = self._get_observation()

        # 记录总利润
        self.total_profit = profit

        # 可选：添加惩罚项，避免频繁交易等
        return next_state, reward, done, {}

    def render(self, mode='human'):
        profit = self.net_worth - self.initial_balance
        print(f'Step: {self.current_step}, Balance: {self.balance:.2f}, Position: {self.position:.4f}, Net Worth: {self.net_worth:.2f}, Profit: {profit:.2f}')
