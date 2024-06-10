# Confidential, Copyright 2020, Sony Corporation of America, All rights reserved.

import pandemic_simulator as ps


def test_register_persons() -> None:
    ps.init_globals()
    cr = ps.env.globals.registry
    assert cr
    config = ps.env.PandemicSimConfig(num_persons=10,
                                      location_configs=[
                                          ps.env.LocationConfig(ps.env.Home, 5),
                                          ps.env.LocationConfig(ps.env.Office, 2, 5),
                                          ps.env.LocationConfig(ps.env.School, 1)
                                      ])
    ps.env.make_locations(config)
    home_id = cr.location_ids_of_type(ps.env.Home)[0]
    school_id = cr.location_ids_of_type(ps.env.School)[0]
    office_id = cr.location_ids_of_type(ps.env.Office)[0]

    m = ps.env.Minor(ps.env.PersonID('minor_0', 3), home_id, school=school_id)
    a = ps.env.Worker(ps.env.PersonID('adult_0', 36), home_id, work=office_id)

    assert (m.id in cr.get_persons_in_location(home_id))
    assert (a.id in cr.get_persons_in_location(home_id))
