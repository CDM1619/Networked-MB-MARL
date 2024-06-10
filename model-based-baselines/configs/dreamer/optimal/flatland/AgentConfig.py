from configs.Config import Config


class DreamerConfig(Config):
    def __init__(self):
        super().__init__()
        self.HIDDEN = 400
        self.MODEL_HIDDEN = 400
        self.EMBED = 400
        self.N_CATEGORICALS = 32
        self.N_CLASSES = 32
        self.STOCHASTIC = self.N_CATEGORICALS * self.N_CLASSES
        self.DETERMINISTIC = 600
        self.FEAT = self.STOCHASTIC + self.DETERMINISTIC
        self.GLOBAL_FEAT = self.FEAT + self.EMBED
        self.VALUE_LAYERS = 2
        self.VALUE_HIDDEN = 400
        self.PCONT_LAYERS = 2
        self.PCONT_HIDDEN = 400
        self.ACTION_SIZE = 3
        self.ACTION_LAYERS = 2
        self.ACTION_HIDDEN = 400
        self.REWARD_LAYERS = 2
        self.REWARD_HIDDEN = 400
        self.GAMMA = 0.99
        self.DISCOUNT = 0.99
        self.DISCOUNT_LAMBDA = 0.95
        self.IN_DIM = 205
