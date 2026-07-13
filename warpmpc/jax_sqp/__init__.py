"""Fixed-pattern sparse MPC SQP tools."""

from .casadi_stage import CasadiStageFunction
from .line_search import (
    FilterLineSearchResult,
    FilterLineSearchSettings,
    LINE_SEARCH_CONSTRAINT,
    LINE_SEARCH_COST,
    LINE_SEARCH_COST_OR_CONSTRAINT,
    LINE_SEARCH_MAX_ITER,
    constraint_violation,
    filter_line_search_from_evaluations,
    make_step_lengths,
)
from .sparse_mpc import MPAXSettings, SparseMPCProblem, build_sparse_mpc_plan, compile_sparse_mpc_sqp
from .types import (
    CompiledSparseMPCSQP,
    SQPLinearization,
    SQPLineSearchStepResult,
    SQPSolveResult,
    SQPStepResult,
    SparseMPCPlan,
    OSQPWarmStart,
)

__all__ = [
    "CasadiStageFunction",
    "CompiledSparseMPCSQP",
    "FilterLineSearchResult",
    "FilterLineSearchSettings",
    "LINE_SEARCH_CONSTRAINT",
    "LINE_SEARCH_COST",
    "LINE_SEARCH_COST_OR_CONSTRAINT",
    "LINE_SEARCH_MAX_ITER",
    "MPAXSettings",
    "SQPLinearization",
    "SQPLineSearchStepResult",
    "SQPSolveResult",
    "SQPStepResult",
    "SparseMPCPlan",
    "OSQPWarmStart",
    "SparseMPCProblem",
    "build_sparse_mpc_plan",
    "compile_sparse_mpc_sqp",
    "constraint_violation",
    "filter_line_search_from_evaluations",
    "make_step_lengths",
]
