# Confidential, Copyright 2020, Sony Corporation of America, All rights reserved.

import pandemic_simulator as ps


def test_job_counselor() -> None:
    ps.init_globals()
    cr = ps.env.globals.registry
    assert cr

    config = ps.env.PandemicSimConfig(num_persons=10,
                                      location_configs=[
                                          ps.env.LocationConfig(ps.env.GroceryStore, 1, 3),
                                          ps.env.LocationConfig(ps.env.Office, 2, 5)
                                      ])

    ps.env.make_locations(config)

    job_counselor = ps.env.JobCounselor(config.location_configs)

    all_work_ids = cr.location_ids_of_type(ps.env.BusinessBaseLocation)
    total_jobs = sum([config.num * config.num_assignees for config in config.location_configs])

    for _ in range(total_jobs):
        work_package = job_counselor.next_available_work()
        assert work_package
        assert work_package.work in all_work_ids

    # should return None when asked for next available work id since all are taken
    assert job_counselor.next_available_work() is None
