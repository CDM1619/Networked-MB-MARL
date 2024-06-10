from agent.controllers.DreamerController import DreamerController
from configs.dreamer.DreamerAgentConfig import DreamerConfig


class DreamerControllerConfig(DreamerConfig):
    def __init__(self):
        super().__init__()
        self.DEVICE = 'cuda'
        self.EXPL_DECAY = 0.9999
        self.EXPL_NOISE = 0.
        self.EXPL_MIN = 0.

    def create_controller(self):
        return DreamerController(self)
