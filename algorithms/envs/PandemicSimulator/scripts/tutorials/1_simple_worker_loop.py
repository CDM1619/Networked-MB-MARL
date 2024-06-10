# Confidential, Copyright 2020, Sony Corporation of America, All rights reserved.

from tqdm import trange

import pandemic_simulator as ps


def simple_worker_loop() -> None:
    """A simple worker loop tutorial, where a person goes to an assigned office during work time and goes back home
    after work."""
    print('\nSimple worker loop tutorial', flush=True)

    # the first thing to do at the start of any experiment is to initialize a few global parameters
    # these parameters are shared across the entire repo
    ps.init_globals(
        seed=0,  # if None, the experiment is not seeded and would initialized differently each time
        registry=None,  # if None, a registry is created and used
        # a registry does bookkeeping of all people and locations used in the experiment
    )

    # init locations
    home = ps.env.Home()
    work = ps.env.Office()  # any subclass of BusinessLocation can be a workplace, e.g. Bar, Restaurant, Hospital, etc.

    # init a worker
    person = ps.env.Worker(
        person_id=ps.env.PersonID('worker', age=35),  # person_id is a unique id for this person
        home=home.id,  # specify the home_id that person is assigned to
        work=work.id,  # specify the id of the person's workplace
    )

    # Init simulator
    sim = ps.env.PandemicSim(
        locations=[work, home],  # a list of all locations
        persons=[person]  # a list of all persons
    )
    # PandemicSim by default creates and uses randomized testing and an SEIR infection model

    # Iterate through steps in the simulator, where each step advances an hour
    for _ in trange(24, desc='Simulating hour'):
        sim.step()

    # Or iterate by advancing in days by calling step_day in the simulator
    for _ in trange(10, desc='Simulating day'):
        sim.step_day()

    # The above loop iterates the simulator with no movement restrictions
    # To impose restrictions, for example, Stage-2 of austin_regulations
    sim.impose_regulation(ps.sh.austin_regulations[2])

    # Calling step_day now will run the simulator under Stage-2 regulation
    for _ in trange(10, desc='Simulating day (Under Stage-2)'):
        sim.step_day()


if __name__ == '__main__':
    simple_worker_loop()
