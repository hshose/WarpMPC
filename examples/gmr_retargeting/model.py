"""GMR's two-stage manifold IK, compiled to sparse WarpMPC QPs.

MuJoCo/Mink are used only for model loading and the independent CPU reference.
The timed GPU path evaluates kinematics, SE(3) residuals, sparse Gauss--Newton
matrices, ADMM and quaternion integration entirely on the device.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import pathlib
import sys
import types
from dataclasses import dataclass

import casadi as ca
import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import scipy.sparse as sp

from warpmpc.jax_osqp import OSQPSettings, build_osqp_plan, compile_osqp
from warpmpc.numpysadi import convert

ROOT = pathlib.Path(__file__).resolve().parents[2]


def load_gmr(gmr_root: pathlib.Path, height: float | None = None):
    """Load the original class without importing its optional GUI/Torch modules."""
    package = "_warpmpc_gmr_reference"
    directory = gmr_root.resolve() / "general_motion_retargeting"
    if not (directory / "motion_retarget.py").is_file():
        raise FileNotFoundError(f"GMR source missing in {directory}")
    module = types.ModuleType(package)
    module.__path__ = [str(directory)]
    sys.modules[package] = module
    name = package + ".motion_retarget"
    spec = importlib.util.spec_from_file_location(name, directory / "motion_retarget.py")
    ref = importlib.util.module_from_spec(spec)
    sys.modules[name] = ref
    spec.loader.exec_module(ref)
    with contextlib.redirect_stdout(io.StringIO()):
        return ref.GeneralMotionRetargeting(
            "smplx",
            "unitree_g1",
            actual_human_height=height,
            solver="daqp",
            damping=0.5,
            verbose=False,
        )


def quat_mul(a, b):
    return ca.vertcat(
        a[0] * b[0] - ca.dot(a[1:], b[1:]), a[0] * b[1:] + b[0] * a[1:] + ca.cross(a[1:], b[1:])
    )


def quat_matrix(q):
    v = q[1:]
    return (q[0] ** 2 - ca.dot(v, v)) * ca.SX.eye(3) + 2 * v @ v.T + 2 * q[0] * ca.skew(v)


def pose_error(pos, quat, target):
    """log(T_current^-1 T_target), Mink FrameTask's body-frame error."""
    inverse = ca.vertcat(quat[0], -quat[1:])
    relative = quat_mul(inverse, target[3:7])
    relative = ca.if_else(relative[0] < 0, -relative, relative)
    v, w = relative[1:], relative[0]
    s2 = ca.dot(v, v)
    # Safe inactive branches matter for the derivative and GPU select code.
    s = ca.sqrt(ca.fmax(s2, 1e-16))
    theta = 2 * ca.atan2(s, w)
    factor = ca.if_else(s2 < 1e-8, 2 + s2 / 3 + 3 * s2 * s2 / 20, theta / s)
    phi = factor * v
    theta2 = ca.dot(phi, phi)
    alpha = ca.if_else(
        s2 < 1e-6,
        1 / 12 + theta2 / 720 + theta2**2 / 30240,
        (1 - theta * w / (2 * s)) / ca.fmax(theta2, 1e-16),
    )
    p = quat_matrix(quat).T @ (target[:3] - pos)
    r = p - ca.cross(phi, p) / 2 + alpha * ca.cross(phi, ca.cross(phi, p))
    return ca.vertcat(r, phi)


@dataclass(frozen=True)
class SolverSettings:
    sqp_iterations: int = 2  # Per stage; two GMR stages.
    admm_iterations: int = 50
    rho: float = 0.1
    sigma: float = 1e-6
    alpha: float = 1.6
    backend: str = "warp"
    derivative_backend: str = "warp"
    scaled: bool = True
    adaptive_stopping: bool = False  # Original GMR error-decrease rule.
    segment_budget: int = 128


class RetargetModel:
    def __init__(self, gmr):
        self.gmr = gmr
        self.mj = m = gmr.model
        if m.nq != m.nv + 1 or m.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError("This example supports one floating base and hinge joints")
        if np.any(m.jnt_type[1:] != mujoco.mjtJoint.mjJNT_HINGE):
            raise ValueError("Non-hinge robot joints are unsupported")
        self.nq, self.nv = m.nq, m.nv
        stages = [gmr.tasks1, gmr.tasks2]
        if not (gmr.use_ik_match_table1 and gmr.use_ik_match_table2):
            raise ValueError("Expected G1's two enabled stages")
        self.names = list(dict.fromkeys(t.frame_name for tasks in stages for t in tasks))
        self.body_ids = [m.body(name).id for name in self.names]
        self.weights = np.zeros((2, len(self.names) * 6))
        self.active = np.zeros((2, len(self.names)))
        for s, tasks in enumerate(stages):
            for task in tasks:
                k = self.names.index(task.frame_name)
                if task.gain != 1 or task.lm_damping != 1:
                    raise ValueError("Expected GMR gain=lm_damping=1")
                self.weights[s, k * 6 : k * 6 + 6] = task.cost
                self.active[s, k] = 1
        self.joint_ids = np.flatnonzero(m.jnt_limited)
        self.dofs = m.jnt_dofadr[self.joint_ids].copy()
        self.qindices = m.jnt_qposadr[self.joint_ids].copy()
        self.lower = m.jnt_range[self.joint_ids, 0].copy()
        self.upper = m.jnt_range[self.joint_ids, 1].copy()
        self.limit_gain = gmr.ik_limits[0].gain
        self.human_names = list(gmr.human_scale_table)
        self._build_symbolic()

    def targets(self):
        result = np.zeros((2, len(self.names), 7))
        result[:, :, 3] = 1
        for s, tasks in enumerate((self.gmr.tasks1, self.gmr.tasks2)):
            for task in tasks:
                T = task.transform_target_to_world
                k = self.names.index(task.frame_name)
                result[s, k] = np.r_[T.translation(), T.rotation().wxyz]
        return result.reshape(2, -1)

    def pack_human(self, frames):
        return np.array(
            [
                [np.r_[frame[name][0], frame[name][1]] for name in self.human_names]
                for frame in frames
            ]
        )

    def preprocess(self, human):
        """Device counterpart of original GMR.update_targets (no ground snapping)."""
        root = self.human_names.index(self.gmr.human_root_name)
        scale = jnp.asarray(
            [self.gmr.human_scale_table[n] for n in self.human_names], dtype=human.dtype
        )
        root_pos = human[:, root : root + 1, :3]
        pos = (human[:, :, :3] - root_pos) * scale[None, :, None] + root_pos * scale[root]
        offset_q = jnp.asarray(
            [self.gmr.rot_offsets1[n].as_quat(scalar_first=True) for n in self.human_names],
            dtype=human.dtype,
        )
        rotation = jquat_mul(human[:, :, 3:], offset_q[None])
        offset_p = jnp.asarray(
            [self.gmr.pos_offsets1[n] for n in self.human_names], dtype=human.dtype
        )
        vector = jnp.broadcast_to(offset_p[None], pos.shape)
        cross = 2 * jnp.cross(rotation[:, :, 1:], vector)
        pos += vector + rotation[:, :, :1] * cross + jnp.cross(rotation[:, :, 1:], cross)
        pos = pos - jnp.asarray([0.0, 0.0, self.gmr.ground_offset], dtype=human.dtype)
        poses = jnp.concatenate((pos, rotation), axis=-1)
        stages = []
        for table in (self.gmr.ik_match_table1, self.gmr.ik_match_table2):
            indices = [self.human_names.index(table[n][0]) for n in self.names]
            stages.append(poses[:, indices].reshape(human.shape[0], -1))
        return jnp.stack(stages, axis=1)

    def _build_symbolic(self):
        m = self.mj
        q = ca.SX.sym("q", m.nq)
        target = ca.SX.sym("target", 7 * len(self.names))
        weight = ca.SX.sym("weight", 6 * len(self.names))
        positions, rotations = [ca.SX.zeros(3)], [ca.DM([1, 0, 0, 0])]
        ancestors = [[]]
        axes, anchors = {}, {}
        for body in range(1, m.nbody):
            parent = m.body_parentid[body]
            rp = rotations[parent]
            pos = positions[parent] + quat_matrix(rp) @ ca.DM(m.body_pos[body])
            rot = quat_mul(rp, ca.DM(m.body_quat[body]))
            chain = list(ancestors[parent])
            for joint in range(m.body_jntadr[body], m.body_jntadr[body] + m.body_jntnum[body]):
                adr = m.jnt_qposadr[joint]
                if m.jnt_type[joint] == mujoco.mjtJoint.mjJNT_FREE:
                    pos, rot = q[:3], q[3:7]
                else:
                    axes[joint] = quat_matrix(rot) @ ca.DM(m.jnt_axis[joint])
                    anchors[joint] = pos + quat_matrix(rot) @ ca.DM(m.jnt_pos[joint])
                    chain.append(joint)
                    angle = q[adr] - m.qpos0[adr]
                    delta = ca.vertcat(
                        ca.cos(angle / 2), ca.DM(m.jnt_axis[joint]) * ca.sin(angle / 2)
                    )
                    updated = quat_mul(rot, delta)
                    pivot = ca.DM(m.jnt_pos[joint])
                    pos += (quat_matrix(rot) - quat_matrix(updated)) @ pivot
                    rot = updated
            positions.append(pos)
            rotations.append(rot)
            ancestors.append(chain)
        # Differentiate the six-dimensional local pose error once, then compose
        # with the sparse geometric Jacobian. Differentiating every FK expression
        # through every joint makes unnecessarily large generated GPU kernels.
        local = ca.SX.sym("local", 7)
        relative = ca.SX.sym("relative", 7)
        local_error = pose_error(local[:3], local[3:], relative)
        local_jac = ca.jacobian(local_error, local)
        at_identity = ca.DM([0, 0, 0, 1, 0, 0, 0])
        local_jac = ca.substitute(local_jac, local, at_identity)
        local_jac = ca.horzcat(local_jac[:, :3], local_jac[:, 4:7] / 2)
        error_jac = ca.Function(
            "local_pose_error_jac",
            [relative],
            [ca.substitute(local_error, local, at_identity), ca.sparsify(local_jac)],
        )
        task_errors, task_jacs = [], []
        base_rotation = quat_matrix(q[3:7])
        for i, b in enumerate(self.body_ids):
            rotation = quat_matrix(rotations[b])
            inverse = ca.vertcat(rotations[b][0], -rotations[b][1:])
            rel = ca.vertcat(
                rotation.T @ (target[7 * i : 7 * i + 3] - positions[b]),
                quat_mul(inverse, target[7 * i + 3 : 7 * i + 7]),
            )
            error, differential = error_jac(rel)
            jp, jr = ca.SX(3, m.nv), ca.SX(3, m.nv)
            jp[:, :3] = ca.SX.eye(3)
            jp[:, 3:6] = -ca.skew(positions[b] - q[:3]) @ base_rotation
            jr[:, 3:6] = base_rotation
            for joint in ancestors[b]:
                dof = int(m.jnt_dofadr[joint])
                jp[:, dof] = ca.cross(axes[joint], positions[b] - anchors[joint])
                jr[:, dof] = axes[joint]
            geometric = ca.vertcat(rotation.T @ jp, rotation.T @ jr)
            task_errors.append(error)
            task_jacs.append(ca.sparsify(differential @ geometric))
        errors = ca.vertcat(*task_errors)
        J = ca.vertcat(*task_jacs)
        weighted = ca.times(weight, errors)
        Jw = ca.diag(weight) @ J
        H = ca.sparsify(
            Jw.T @ Jw + (self.gmr.damping + ca.dot(weighted, weighted)) * ca.SX.eye(m.nv)
        )
        c = Jw.T @ weighted
        rows, cols = ca.triu(H).sparsity().get_triplet()
        self.rows, self.cols = np.asarray(rows), np.asarray(cols)
        self.diag_indices = np.flatnonzero(self.rows == self.cols)
        self.pattern = sp.csc_matrix((np.ones(len(rows)), (rows, cols)), shape=(m.nv, m.nv))
        self.qp_function = ca.Function(
            "gmr_qp",
            [q, target, weight],
            [
                ca.densify(ca.vertcat(*[H[r, c_] for r, c_ in zip(rows, cols)])),
                ca.densify(c),
            ],
        )
        self.error_function = ca.Function("gmr_error", [q, target], [ca.densify(errors)])
        self.jacobian_function = ca.Function("gmr_jacobian", [q, target], [ca.densify(J)])
        self.fk_function = ca.Function(
            "gmr_fk",
            [q],
            [
                ca.densify(
                    ca.vertcat(*[ca.vertcat(positions[b], rotations[b]) for b in self.body_ids])
                )
            ],
        )
        print(
            f"GMR: nq={m.nq}, nv={m.nv}, tasks={len(self.names)}, nnz(P)={len(rows)}, "
            f"QP instructions={self.qp_function.n_instructions()}",
            flush=True,
        )

    def compile_primitives(self, settings: SolverSettings):
        """One batched QP step; stage may be a scalar or independent per lane."""
        if settings.sqp_iterations < 1 or settings.admm_iterations < 1 or settings.rho <= 0:
            raise ValueError("Iteration counts and rho must be positive")
        a = sp.csc_matrix(
            (np.ones(len(self.dofs)), (np.arange(len(self.dofs)), self.dofs)),
            shape=(len(self.dofs), self.nv),
        )
        plan = build_osqp_plan(
            self.pattern,
            a,
            -np.ones(len(self.dofs)),
            np.ones(len(self.dofs)),
            OSQPSettings(
                max_iter=settings.admm_iterations,
                rho=settings.rho,
                sigma=settings.sigma,
                alpha=settings.alpha,
                scaling=0,
            ),
        )
        solver = compile_osqp(
            plan,
            dtype=np.float32,
            qdldl_backend=settings.backend,
            transpose_work=True,
            segmented=True,
            segment_budget=settings.segment_budget,
        )
        qp_one = convert(self.qp_function, backend=settings.derivative_backend)
        qp = jax.vmap(lambda q, t, w: tuple(x[:, 0] for x in qp_one(q, t, w)))
        err_one = convert(self.error_function, backend=settings.derivative_backend)
        errors = jax.vmap(lambda q, t: err_one(q, t)[:, 0])
        weights = jnp.asarray(self.weights, dtype=jnp.float32)
        active = jnp.asarray(np.repeat(self.active, 6, axis=1), dtype=jnp.float32)

        def step(q, target, stage):
            w = jnp.broadcast_to(weights[stage], (q.shape[0], self.weights.shape[1]))
            p, c = qp(q, target, w)
            d = (
                jax.lax.rsqrt(jnp.maximum(p[:, self.diag_indices], 1e-12))
                if settings.scaled
                else jnp.ones_like(c)
            )
            p = p * d[:, self.rows] * d[:, self.cols]
            c = c * d
            lower = (
                self.limit_gain * (jnp.asarray(self.lower) - q[:, self.qindices]) / d[:, self.dofs]
            )
            upper = (
                self.limit_gain * (jnp.asarray(self.upper) - q[:, self.qindices]) / d[:, self.dofs]
            )
            dx = solver.solve(p, jnp.ones((1, len(self.dofs)), q.dtype), c, lower, upper)[0] * d
            return integrate(q, dx)

        return step, errors, active, solver

    def compile(self, settings: SolverSettings, *, diagnostics=False):
        step, errors, active, solver = self.compile_primitives(settings)

        def retarget(q, targets):
            counts, decreases = [], []
            for stage in range(2):
                target = targets[:, stage]
                if settings.adaptive_stopping:
                    initial = jnp.linalg.norm(errors(q, target) * active[stage], axis=1)

                    def body(_, carry):
                        state, prev, live = carry
                        candidate = step(state, target, stage)
                        error = jnp.linalg.norm(errors(candidate, target) * active[stage], axis=1)
                        return (
                            jnp.where(live[:, None], candidate, state),
                            error,
                            live & (prev - error > 0.001),
                        )

                    if diagnostics:

                        def traced_body(i, carry):
                            state, previous, live, count, last = carry
                            candidate, error, continuing = body(i, (state, previous, live))
                            return (
                                candidate,
                                error,
                                continuing,
                                count + live.astype(jnp.int32),
                                jnp.where(live, previous - error, last),
                            )

                        q, _, _, count, decrease = jax.lax.fori_loop(
                            0,
                            settings.sqp_iterations,
                            traced_body,
                            (
                                q,
                                initial,
                                jnp.ones(q.shape[0], dtype=bool),
                                jnp.zeros(q.shape[0], dtype=jnp.int32),
                                jnp.zeros(q.shape[0], dtype=q.dtype),
                            ),
                        )
                        counts.append(count)
                        decreases.append(decrease)
                    else:
                        q, _, _ = jax.lax.fori_loop(
                            0,
                            settings.sqp_iterations,
                            body,
                            (q, initial, jnp.ones(q.shape[0], dtype=bool)),
                        )
                else:
                    q = jax.lax.fori_loop(
                        0, settings.sqp_iterations, lambda _, state: step(state, target, stage), q
                    )
                    if diagnostics:
                        counts.append(
                            jnp.full(q.shape[0], settings.sqp_iterations, dtype=jnp.int32)
                        )
                        decreases.append(jnp.zeros(q.shape[0], dtype=q.dtype))
            return (
                (q, jnp.stack(counts, axis=-1), jnp.stack(decreases, axis=-1)) if diagnostics else q
            )

        return jax.jit(retarget), solver


def jquat_mul(a, b):
    return jnp.concatenate(
        (
            a[..., :1] * b[..., :1] - jnp.sum(a[..., 1:] * b[..., 1:], axis=-1, keepdims=True),
            a[..., :1] * b[..., 1:] + b[..., :1] * a[..., 1:] + jnp.cross(a[..., 1:], b[..., 1:]),
        ),
        axis=-1,
    )


def integrate(q, dx):
    """MuJoCo freejoint integration: world translation, local angular velocity."""
    v = dx[:, 3:6]
    angle = jnp.linalg.norm(v, axis=-1, keepdims=True)
    dv = v * (0.5 * jnp.sinc(angle / (2 * jnp.pi)))
    dw = jnp.cos(angle / 2)
    w, xyz = q[:, 3:4], q[:, 4:7]
    quat = jnp.concatenate(
        (w * dw - jnp.sum(xyz * dv, axis=1, keepdims=True), w * dv + dw * xyz + jnp.cross(xyz, dv)),
        axis=1,
    )
    quat /= jnp.linalg.norm(quat, axis=1, keepdims=True)
    return jnp.concatenate((q[:, :3] + dx[:, :3], quat, q[:, 7:] + dx[:, 6:]), axis=1)
