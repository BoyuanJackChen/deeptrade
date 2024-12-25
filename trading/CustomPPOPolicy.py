# trading/CustomPPOPolicy.py
from stable_baselines3.common.policies import ActorCriticPolicy
from TradingFeatureExtractor import TransformerFeatureExtractor

class CustomPPOPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPPOPolicy, self).__init__(*args, **kwargs,
            features_extractor_class=TransformerFeatureExtractor,
            features_extractor_kwargs=dict(
                embed_dim=64,
                num_heads=4,
                ff_dim=128,
                num_layers=2,
                window_size=50,
                feature_dim=5
            )
        )
