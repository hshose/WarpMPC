#!/usr/bin/env python3
"""Offline independent flight-law parity check for the C++ observer checker.

Usage: .venv/bin/python cpp/scripts/check_ball_estimator_parity.py CHECK_BINARY
The executable also checks causal impacts, finite arm inertia, delayed updates,
marker identity/loss, and release detection. No hardware or stream is opened.
"""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile
import sys

import numpy as np
from scipy.integrate import solve_ivp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="juggling_cpp_observer_") as temp:
        output = Path(temp) / "flight.txt"
        subprocess.run([str(args.binary.resolve()), str(output)], check=True)
        cases = np.loadtxt(output)
        contact = np.loadtxt(str(output)+".contact")
    initial = np.array([0., 0., 1., .12, -.04, .3, 0., 0., 0., .02, -.01, .04])
    gravity = np.array([.0928822557147292, -.010164049455359378, -9.676561546699691])
    errors = []
    for row in cases:
        model, dt = row[:2]
        linear_drag = .09093571620319228
        quadratic_drag = .06386175735959494 if model else 0.
        def rhs(_t, state):
            velocity = state[3:6]
            acceleration = gravity - (linear_drag + quadratic_drag * np.sqrt(velocity @ velocity + 1e-10)) * velocity + state[9:12]
            return np.r_[velocity, acceleration, np.zeros(3), -state[9:12]]
        if dt:
            reference = solve_ivp(rhs, (0., dt), initial, rtol=1e-12, atol=1e-14).y[:, -1]
        else:
            reference = initial
        errors.append(float(np.max(np.abs(reference-row[2:]))))
    maximum = max(errors)
    # The preserved Python RK4 law subdivides at 20 ms: 0.1 micrometre
    # bounds accumulated error over these deliberately long 0.15 s intervals.
    if maximum > 1e-7:
        raise RuntimeError(f"C++ flight differs from independent high-accuracy ODE solution: {maximum:g}")
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from arm.spin_contact import impact
    normal, relative, observed_velocity, observed_spin = contact.reshape(4, 3)
    tangent = np.cross(normal, [1., 0., 0.]); tangent /= np.linalg.norm(tangent)
    tangents = np.column_stack([tangent, np.cross(normal, tangent)])
    inverse_mass = np.array([[.7, .12, -.08], [.12, .9, .07], [-.08, .07, 1.1]])
    jacobian = np.zeros((3, 7)); jacobian[:, :3] = np.linalg.cholesky(inverse_mass)
    velocity, spin, _, _ = impact(normal, tangents, jacobian, np.eye(7), relative,
                                  ball_mass=.0039, radius=.02, restitution=.54,
                                  friction=.3, inertia_ratio=2/3)
    contact_error = float(max(np.max(np.abs(velocity-observed_velocity)), np.max(np.abs(spin-observed_spin))))
    if contact_error > 1e-8:
        raise RuntimeError(f"Finite-arm frictional impulse parity failed: {contact_error:g}")
    print(json.dumps(dict(flight_parity_passed=True, cases=len(cases), maximum_absolute_error=maximum,
                         finite_arm_contact_parity_passed=True, contact_maximum_absolute_error=contact_error)))


if __name__ == "__main__":
    main()
