"""Prediction target selectors for AMPC imitation."""

from __future__ import annotations

from typing import NamedTuple

FIRST_ACTION = 0
ACTION_SEQUENCE = 1
PRIMAL_SOLUTION = 2


class OutputSpec(NamedTuple):
    target: int
    nx: int
    nu: int
    nz: int
    horizon_steps: int
    target_dim: int


def make_output_spec(target: str, *, nx: int, nu: int, nz: int, horizon_steps: int) -> OutputSpec:
    normalized = target.strip().lower().replace("-", "_")
    if normalized in ("first_action", "action", "first_input", "input"):
        target_id = FIRST_ACTION
        target_dim = nu
    elif normalized in ("action_sequence", "input_sequence", "open_loop_actions", "open_loop_inputs"):
        target_id = ACTION_SEQUENCE
        target_dim = horizon_steps * nu
    elif normalized in ("primal", "primal_solution", "solution_vector", "z"):
        target_id = PRIMAL_SOLUTION
        target_dim = (horizon_steps + 1) * nz
    else:
        raise ValueError(
            "prediction target must be one of first_action, action_sequence, or primal_solution"
        )
    return OutputSpec(
        target=target_id,
        nx=nx,
        nu=nu,
        nz=nz,
        horizon_steps=horizon_steps,
        target_dim=target_dim,
    )


def prediction_target_name(spec: OutputSpec) -> str:
    if spec.target == FIRST_ACTION:
        return "first_action"
    if spec.target == ACTION_SEQUENCE:
        return "action_sequence"
    if spec.target == PRIMAL_SOLUTION:
        return "primal_solution"
    raise ValueError(f"unknown prediction target id {spec.target}")


def select_prediction_target(spec: OutputSpec, primal_solution):
    """Extract the supervised target from an MPC primal solution vector."""

    stages = primal_solution.reshape((primal_solution.shape[0], spec.horizon_steps + 1, spec.nz))
    if spec.target == FIRST_ACTION:
        return stages[:, 0, spec.nx : spec.nx + spec.nu]
    if spec.target == ACTION_SEQUENCE:
        actions = stages[:, : spec.horizon_steps, spec.nx : spec.nx + spec.nu]
        return actions.reshape((primal_solution.shape[0], spec.target_dim))
    if spec.target == PRIMAL_SOLUTION:
        return primal_solution
    raise ValueError(f"unknown prediction target id {spec.target}")


def first_action_from_prediction(spec: OutputSpec, prediction):
    """Map any supported prediction target back to the action applied in MPC."""

    if spec.target == FIRST_ACTION:
        return prediction
    if spec.target == ACTION_SEQUENCE:
        return prediction[:, : spec.nu]
    if spec.target == PRIMAL_SOLUTION:
        stages = prediction.reshape((prediction.shape[0], spec.horizon_steps + 1, spec.nz))
        return stages[:, 0, spec.nx : spec.nx + spec.nu]
    raise ValueError(f"unknown prediction target id {spec.target}")
