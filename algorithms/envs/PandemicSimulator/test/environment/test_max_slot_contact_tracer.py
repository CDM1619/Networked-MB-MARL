# Confidential, Copyright 2020, Sony Corporation of America, All rights reserved.

import numpy as np
import pytest
from orderedset import OrderedSet

from pandemic_simulator.environment import MaxSlotContactTracer, PersonID


@pytest.fixture
def contact_tracer() -> MaxSlotContactTracer:
    return MaxSlotContactTracer(storage_slots=5, time_slot_scale=24)


def test_emtpy_contact_tracer(contact_tracer: MaxSlotContactTracer) -> None:
    assert len(contact_tracer.get_contacts(PersonID('a', 10))) == 0


def test_contacts(contact_tracer: MaxSlotContactTracer) -> None:
    p1 = PersonID('a', 30)
    p2 = PersonID('b', 40)
    p3 = PersonID('c', 50)
    p4 = PersonID('d', 50)

    contacts = OrderedSet([(p1, p2), (p2, p4)])
    contact_tracer.add_contacts(contacts)

    traces = contact_tracer.get_contacts(p1)
    assert len(traces) == 1
    assert p2 in traces
    np.testing.assert_array_almost_equal(traces[p2], [1. / 24., 0., 0., 0., 0.])

    traces = contact_tracer.get_contacts(p2)
    assert len(traces) == 2
    assert p1 in traces and p4 in traces
    np.testing.assert_array_almost_equal(traces[p1], [1. / 24., 0., 0., 0., 0.])
    np.testing.assert_array_almost_equal(traces[p4], [1. / 24., 0., 0., 0., 0.])

    traces = contact_tracer.get_contacts(p4)
    assert len(traces) == 1
    assert p2 in traces
    np.testing.assert_array_almost_equal(traces[p2], [1. / 24., 0., 0., 0., 0.])

    contacts = OrderedSet([(p1, p2), (p3, p4)])
    contact_tracer.add_contacts(contacts)

    traces = contact_tracer.get_contacts(p1)
    assert len(traces) == 1
    assert p2 in traces
    np.testing.assert_array_almost_equal(traces[p2], [2. / 24., 0., 0., 0., 0.])

    traces = contact_tracer.get_contacts(p2)
    assert len(traces) == 2
    assert p1 in traces and p4 in traces
    np.testing.assert_array_almost_equal(traces[p1], [2. / 24., 0., 0., 0., 0.])
    np.testing.assert_array_almost_equal(traces[p4], [1. / 24., 0., 0., 0., 0.])

    traces = contact_tracer.get_contacts(p4)
    assert len(traces) == 2
    assert p2 in traces and p3 in traces
    np.testing.assert_array_almost_equal(traces[p2], [1. / 24., 0., 0., 0., 0.])
    np.testing.assert_array_almost_equal(traces[p3], [1. / 24., 0., 0., 0., 0.])

    contact_tracer.new_time_slot()

    traces = contact_tracer.get_contacts(p2)
    assert len(traces) == 2
    assert p1 in traces and p4 in traces
    np.testing.assert_array_almost_equal(traces[p1], [0., 2. / 24., 0., 0., 0.])
    np.testing.assert_array_almost_equal(traces[p4], [0., 1. / 24., 0., 0., 0.])

    contacts = OrderedSet([(p1, p2)])
    contact_tracer.add_contacts(contacts)

    traces = contact_tracer.get_contacts(p2)
    assert len(traces) == 2
    assert p1 in traces and p4 in traces
    np.testing.assert_array_almost_equal(traces[p1], [1 / 24., 2. / 24., 0., 0., 0.])
    np.testing.assert_array_almost_equal(traces[p4], [0., 1. / 24., 0., 0., 0.])


def test_contact_removal(contact_tracer: MaxSlotContactTracer) -> None:
    p1 = PersonID('a', 30)
    p2 = PersonID('b', 40)
    p3 = PersonID('c', 50)

    contacts = OrderedSet([(p1, p2), (p2, p3)])
    contact_tracer.add_contacts(contacts)

    traces = contact_tracer.get_contacts(p1)
    assert len(traces) == 1
    assert p2 in traces
    np.testing.assert_array_almost_equal(traces[p2], [1. / 24., 0., 0., 0., 0.])

    traces = contact_tracer.get_contacts(p2)
    assert len(traces) == 2
    assert p1 in traces and p3 in traces
    np.testing.assert_array_almost_equal(traces[p1], [1. / 24., 0., 0., 0., 0.])
    np.testing.assert_array_almost_equal(traces[p3], [1. / 24., 0., 0., 0., 0.])

    for _ in range(5):
        contact_tracer.new_time_slot()
        contacts = OrderedSet([(p2, p3)])
        contact_tracer.add_contacts(contacts)

    traces = contact_tracer.get_contacts(p1)
    assert len(traces) == 0
    traces = contact_tracer.get_contacts(p2)
    assert len(traces) == 1
    assert p3 in traces
    np.testing.assert_array_almost_equal(traces[p3], [1. / 24., 1. / 24., 1. / 24., 1. / 24., 1. / 24.])
