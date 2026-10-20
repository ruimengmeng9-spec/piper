"""Exact replacement of a box payload within an existing merged rigid body.

All inertia matrices are expressed in the link frame, about the stated COM.
The nominal imported flange/locked-gripper/peg composite is preserved at the
nominal 20 g payload and zero grip-depth offset.
"""
import numpy as np


def parallel_axis(mass, displacement):
    r = np.asarray(displacement, dtype=float)
    return mass * (np.dot(r, r) * np.eye(3) - np.outer(r, r))


def box_inertia(mass):
    x, y, z = .01, .01, .10
    return mass / 12 * np.diag([y*y+z*z, x*x+z*z, x*x+y*y])


def replace_payload(total_mass, total_com, total_inertia, peg_mass, axial_offset):
    old_mass = .02
    old_com = np.array([0., 0., .17])
    new_com = old_com + [0., 0., axial_offset]
    # Subtract the nominal payload about the link origin, then add the new one.
    # The URDF explicitly specifies rounded nominal inertia values.
    old_inertia = np.diag([.000017, .000017, .00000033])
    at_origin = total_inertia + parallel_axis(total_mass, total_com)
    at_origin -= old_inertia + parallel_axis(old_mass, old_com)
    new_mass = total_mass - old_mass + peg_mass
    new_total_com = (total_mass * total_com - old_mass * old_com + peg_mass * new_com) / new_mass
    # Retain the imported nominal payload inertia exactly at 20 g; scale the
    # specified box inertia with mass so nominal regression is bit-consistent.
    new_payload_inertia = old_inertia * (peg_mass / old_mass)
    new_total_inertia = at_origin + new_payload_inertia + parallel_axis(peg_mass, new_com)
    new_total_inertia -= parallel_axis(new_mass, new_total_com)
    new_total_inertia = (new_total_inertia + new_total_inertia.T) / 2
    if np.linalg.eigvalsh(new_total_inertia).min() <= 0:
        raise ValueError('Payload replacement produced non-positive inertia')
    return new_mass, new_total_com, new_total_inertia


if __name__ == '__main__':
    # A separate analytical two-body example tests both removing the payload
    # and re-composing translated COMs, rather than copying implementation math.
    base_mass, base_com = .54613114124, np.array([0., 0., .04])
    base_inertia = np.diag([.0007, .0008, .0004])
    old_mass, old_com = .02, np.array([0., 0., .17])
    old_inertia = np.diag([.000017, .000017, .00000033])
    total_mass = base_mass + old_mass
    total_com = (base_mass*base_com + old_mass*old_com) / total_mass
    total_inertia = base_inertia + old_inertia
    total_inertia += parallel_axis(base_mass, base_com-total_com)
    total_inertia += parallel_axis(old_mass, old_com-total_com)
    for peg_mass, offset in [(.02, 0.), (.01, -.01), (.08, .01)]:
        mass, com, inertia = replace_payload(total_mass, total_com, total_inertia, peg_mass, offset)
        expected_com = (base_mass*base_com + peg_mass*(old_com+[0., 0., offset]))/(base_mass+peg_mass)
        expected_inertia = base_inertia + old_inertia*(peg_mass/.02)
        expected_inertia += parallel_axis(base_mass, base_com-expected_com)
        expected_inertia += parallel_axis(peg_mass, old_com+[0., 0., offset]-expected_com)
        assert np.isclose(mass, base_mass+peg_mass)
        assert np.allclose(com, expected_com, atol=1e-12)
        assert np.allclose(inertia, expected_inertia, atol=1e-12)
    print('PAYLOAD_COMPOSITION_TEST_PASSED')
