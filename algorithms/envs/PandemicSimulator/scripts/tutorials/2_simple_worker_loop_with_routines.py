# Confidential, Copyright 2020, Sony Corporation of America, All rights reserved.

from tqdm import trange

import pandemic_simulator as ps


def simple_worker_loop_with_routines() -> None:
    """A simple worker loop tutorial with extra routines to other locations."""
    print('\nSimple worker loop with routines tutorial', flush=True)

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
    restaurant = ps.env.Restaurant()
    store = ps.env.GroceryStore()

    # init a worker
    person = ps.env.Worker(
        person_id=ps.env.PersonID('worker', age=35),  # person_id is a unique id for this person
        home=home.id,  # specify the home_id that person is assigned to
        work=work.id,  # specify the id of the person's workplace
        work_time=work.get_worker_work_time(),  # assign work time for the person
    )

    # setup person routines
    # routines are abstractions that define when, where and how-often a person should transition
    # between locations. A routine is specified using a PersonRoutine dataclass. Here we'll show an
    # example of a during-work routine to visit a restaurant for lunch.
    during_work_routines = [
        ps.env.PersonRoutine(
            start_loc=work.id,
            # the location where this routine can start. Here, we specify it to be the person's workplace id

            end_loc=restaurant.id,
            # the location where the person needs to go. Here, we specify it to be the restaurant

            valid_time=ps.env.SimTimeTuple(hours=tuple(range(11, 14)), week_days=tuple(range(0, 5))),
            # the time when this routine is due to be executed. Here, we specify it to be around noon during weekdays
        )
    ]
    # more routines can be added to the list and the routines will be executed (if and when due) in the order
    # presented in the list.
    person.set_during_work_routines(during_work_routines)

    # similarly, let's add an outside-work routine to visit a grocery store once every week
    outside_work_routines = [
        ps.env.PersonRoutine(
            start_loc=None,  # we specify it as None, so a person can go to the store directly after work if due

            end_loc=store.id,  # store id

            start_trigger=ps.env.SimTimeRoutineTrigger(day=7),
            # notice that we set here start_trigger argument instead of valid_time. start_trigger specifies
            # a trigger to enable the routine. In this case, the routine is triggered every
            # seventh day. Once triggered, it is queued to be executed until it gets triggered again.
            # Advanced tip: SimTimeRoutineTrigger triggers based on sim_time only. If you want to create state
            # based triggers, you can implement it similar to SimTimeRoutineTrigger and use person_state to return
            # a boolean (see test/environment/test_person_routines.py for an example).
        )

    ]
    person.set_outside_work_routines(outside_work_routines)

    # Init simulator
    sim = ps.env.PandemicSim(locations=[work, home, restaurant, store], persons=[person])

    # setup viz to show plots
    viz = ps.viz.SimViz(num_persons=1)

    # Iterate by advancing in days by calling step_day in the simulator
    for _ in trange(10, desc='Simulating day'):
        sim.step_day()
        viz.record(sim.state)

    # show plot
    viz.plot([ps.viz.PlotType.location_assignee_visits, ps.viz.PlotType.location_visitor_visits])


if __name__ == '__main__':
    simple_worker_loop_with_routines()
