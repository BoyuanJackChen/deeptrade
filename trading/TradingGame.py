from Game import Game
from trading.TradingEnv import TradingEnv
import gym

class TradingGame(Game):
    def __init__(self, env):
        self.env = env

    def get_init_board(self):
        state = self.env.reset()
        return state

    def get_next_state(self, state, action):
        next_state, reward, done, _ = self.env.step(action)
        return next_state, reward, done

    def get_valid_moves(self, state):
        # 在交易中所有动作都是有效的
        return [0, 1, 2]

    def get_game_ended(self, state, player):
        # 判断游戏是否结束
        return self.env.current_step >= len(self.env.data)

    def get_canonical_form(self, state, player):
        # 对于单代理环境，不需要转换
        return state

    def get_symmetries(self, board, pi):
        # 交易数据没有对称性，返回原始策略
        return [(board, pi)]
