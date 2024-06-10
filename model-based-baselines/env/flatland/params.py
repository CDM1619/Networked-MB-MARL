from configs.EnvConfigs import FlatlandConfig


def FewAgents(random_seed):
    return FlatlandConfig(height=35,
                          width=25,
                          n_agents=2,
                          n_cities=2,
                          grid_distribution_of_cities=False,
                          max_rails_between_cities=2,
                          max_rail_in_cities=4,
                          observation_builder_config=None,
                          reward_config=None,
                          malfunction_rate=1. / 50,
                          random_seed=random_seed,
                          greedy=True)


def SeveralAgents(random_seed):
    return FlatlandConfig(height=35,
                          width=35,
                          n_agents=5,
                          n_cities=3,
                          grid_distribution_of_cities=False,
                          max_rails_between_cities=2,
                          max_rail_in_cities=4,
                          observation_builder_config=None,
                          reward_config=None,
                          malfunction_rate=1. / 100,
                          random_seed=random_seed,
                          greedy=True)


def PackOfAgents(random_seed):
    return FlatlandConfig(height=35,
                          width=35,
                          n_agents=10,
                          n_cities=4,
                          grid_distribution_of_cities=False,
                          max_rails_between_cities=2,
                          max_rail_in_cities=4,
                          observation_builder_config=None,
                          reward_config=None,
                          malfunction_rate=1. / 150,
                          random_seed=random_seed,
                          greedy=True)


def LotsOfAgents(random_seed):
    return FlatlandConfig(height=40,
                          width=60,
                          n_agents=20,
                          n_cities=6,
                          grid_distribution_of_cities=False,
                          max_rails_between_cities=2,
                          max_rail_in_cities=4,
                          observation_builder_config=None,
                          reward_config=None,
                          malfunction_rate=1. / 200,
                          random_seed=random_seed,
                          greedy=True)
