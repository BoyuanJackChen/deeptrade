from gym.envs.registration import register

register(
    id='TradingEnv-v0',
    entry_point='trading.TradingEnv:TradingEnv',
)
