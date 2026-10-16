"""Robot/world hand-eye calibration. A_T_B maps coordinates B -> A, in metres.

B = robot base, F = panda_link8 (flange), V = Vicon world, P = tracked paddle.
Solve V_T_B @ B_T_F[i] @ F_T_P = V_T_P[i]. The CAD hitting face is NOT P.
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R


def transform(value):
    T = np.asarray(value, float)
    if (T.shape != (4, 4) or not np.isfinite(T).all()
            or not np.allclose(T[3], [0, 0, 0, 1], atol=1e-8)
            or not np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-6)
            or abs(np.linalg.det(T[:3, :3])-1) > 1e-6):
        raise ValueError('Expected a finite, proper SE(3) transform')
    return T


def pose(position, quaternion_xyzw):
    q = np.asarray(quaternion_xyzw, float)
    if q.shape != (4,) or not np.isfinite(q).all() or abs(np.linalg.norm(q)-1) > .01:
        raise ValueError('Invalid measured quaternion')
    T = np.eye(4); T[:3, :3] = R.from_quat(q).as_matrix(); T[:3, 3] = position
    return transform(T)


def inverse(T):
    T = transform(T); out = np.eye(4)
    out[:3, :3] = T[:3, :3].T; out[:3, 3] = -out[:3, :3] @ T[:3, 3]
    return out


def mean_pose(poses):
    poses = np.array([transform(T) for T in poses])
    if len(poses) < 1: raise ValueError('No valid poses')
    out = np.eye(4); out[:3, :3] = R.from_matrix(poses[:, :3, :3]).mean().as_matrix()
    out[:3, 3] = np.median(poses[:, :3, 3], axis=0)
    return out


def errors(expected, observed):
    d = inverse(expected) @ observed
    return np.linalg.norm(d[:3, 3]), np.linalg.norm(R.from_matrix(d[:3, :3]).as_rotvec())


def _pack(T): return np.r_[R.from_matrix(T[:3, :3]).as_rotvec(), T[:3, 3]]


def _unpack(v):
    T = np.eye(4); T[:3, :3] = R.from_rotvec(v[:3]).as_matrix(); T[:3, 3] = v[3:]
    return T


def flange_from_state(state):
    # SDK arrays are COLUMN-major. Calibrate the flange so a Desk EE definition
    # change does not silently change the saved extrinsic.
    B_T_EE = transform(np.array(state['O_T_EE']).reshape(4, 4, order='F'))
    F_T_EE = transform(np.array(state['F_T_EE']).reshape(4, 4, order='F'))
    return B_T_EE @ inverse(F_T_EE)


def solve(samples):
    if len(samples) < 10: raise ValueError('Need at least ten settled, diverse poses')
    A = np.array([transform(s['B_T_F']) for s in samples])
    C = np.array([transform(s['V_T_P']) for s in samples])
    # Pair motions satisfy (inv(Aj) Ai) X = X (inv(Cj) Ci).
    ar, cr, pairs = [], [], []
    for i in range(len(A)):
        for j in range(i):
            a, c = inverse(A[j]) @ A[i], inverse(C[j]) @ C[i]
            av, cv = R.from_matrix(a[:3, :3]).as_rotvec(), R.from_matrix(c[:3, :3]).as_rotvec()
            if np.linalg.norm(av) > .06:
                ar.append(av); cr.append(cv); pairs.append((a, c))
    if len(ar) < 6: raise ValueError('Insufficient rotation excitation')
    singular = np.linalg.svd(ar, compute_uv=False)
    if singular[1] < .15 or singular[1]/singular[0] < .12:
        raise ValueError('Need rotations about at least two independent axes')
    X = np.eye(4); X[:3, :3] = R.align_vectors(ar, cr)[0].as_matrix()
    L = np.vstack([a[:3, :3]-np.eye(3) for a, c in pairs])
    rhs = np.concatenate([X[:3, :3] @ c[:3, 3]-a[:3, 3] for a, c in pairs])
    if np.linalg.cond(L) > 30: raise ValueError('Flange translation is poorly observable')
    X[:3, 3] = np.linalg.lstsq(L, rhs, rcond=None)[0]
    Y = mean_pose([c @ inverse(a @ X) for a, c in zip(A, C)])

    def residual(v, indices):
        x, y = _unpack(v[:6]), _unpack(v[6:]); out = []
        for i in indices:
            d = inverse(y @ A[i] @ x) @ C[i]
            out.extend(d[:3, 3]/.001)
            out.extend(R.from_matrix(d[:3, :3]).as_rotvec()/np.deg2rad(.2))
        return out

    # Fit without every fourth pose, validate withheld poses, then refit all.
    train = [i for i in range(len(A)) if i % 4 != 3]
    holdout = [i for i in range(len(A)) if i % 4 == 3]
    fitted = least_squares(residual, np.r_[_pack(X), _pack(Y)], args=(train,), loss='soft_l1', max_nfev=150)
    Xh, Yh = _unpack(fitted.x[:6]), _unpack(fitted.x[6:])
    validation = np.array([errors(Yh @ A[i] @ Xh, C[i]) for i in holdout])
    if (not fitted.success or np.max(validation[:, 0]) > .005
            or np.max(validation[:, 1]) > np.deg2rad(1)):
        raise ValueError('Held-out calibration residual exceeds 5 mm / 1 degree')
    fitted = least_squares(residual, fitted.x, args=(range(len(A)),), loss='soft_l1', max_nfev=150)
    X, Y = _unpack(fitted.x[:6]), _unpack(fitted.x[6:])
    err = np.array([errors(Y @ a @ X, c) for a, c in zip(A, C)])
    if (not fitted.success or np.max(err[:, 0]) > .005 or np.max(err[:, 1]) > np.deg2rad(1)
            or np.linalg.norm(X[:3, 3]) > .5 or np.linalg.cond(fitted.jac) > 1e4):
        raise ValueError('Calibration failed residual, mount-size or conditioning check')
    return dict(schema_version=1, created_utc=datetime.now(timezone.utc).isoformat(),
                convention='A_T_B maps B coordinates into A; metres; right-handed',
                flange_frame='panda_link8', paddle_subject='hsh_pingpong_paddle',
                F_T_P=X.tolist(), calibration_V_T_B=Y.tolist(),
                world_alignment_is_session_specific=True, samples=len(A),
                translation_rms_m=float(np.sqrt(np.mean(err[:, 0]**2))),
                rotation_rms_deg=float(np.rad2deg(np.sqrt(np.mean(err[:, 1]**2)))),
                max_translation_m=float(err[:, 0].max()),
                holdout_max_translation_m=float(validation[:, 0].max()),
                holdout_max_rotation_deg=float(np.rad2deg(validation[:, 1].max())),
                rotation_excitation_singular_values=singular.tolist(),
                jacobian_condition=float(np.linalg.cond(fitted.jac)))


def load(path, require_live=True):
    data = json.loads(Path(path).read_text())
    if (data.get('schema_version') != 1 or data.get('flange_frame') != 'panda_link8'
            or data.get('paddle_subject') != 'hsh_pingpong_paddle'):
        raise ValueError('Calibration schema/frame/subject mismatch')
    transform(data['F_T_P'])
    if require_live and data.get('source_kind') != 'live_vicon_robot':
        raise ValueError('Hardware startup requires a live Vicon/robot calibration, not synthetic data')
    return data


def align_home(calibration, samples):
    """Fresh, settled measurements at HOME; never reuse calibration-time Y."""
    X = transform(calibration['F_T_P'])
    if len(samples) < 30: raise ValueError('Need at least 30 fresh settled HOME observations')
    candidates = [transform(s['V_T_P']) @ inverse(transform(s['B_T_F']) @ X) for s in samples]
    Y = mean_pose(candidates)
    e = np.array([errors(Y, c) for c in candidates])
    if e[:, 0].max() > .003 or e[:, 1].max() > np.deg2rad(.5):
        raise ValueError('HOME alignment is not stable (3 mm / 0.5 degree)')
    # Model gravity is -base Z. A tilted/mis-mounted base needs a gravity-aware
    # model; world yaw/translation changes are supported without recompilation.
    if np.arccos(np.clip(Y[2, 2], -1, 1)) > np.deg2rad(3):
        raise ValueError('Robot base Z is not aligned with Vicon up within 3 degrees')
    return dict(schema_version=1, created_utc=datetime.now(timezone.utc).isoformat(),
                V_T_B=Y.tolist(), B_T_V=inverse(Y).tolist(), samples=len(samples),
                fixed_calibration_sha256=hashlib.sha256(json.dumps(calibration, sort_keys=True).encode()).hexdigest(),
                max_translation_scatter_m=float(e[:, 0].max()),
                max_rotation_scatter_deg=float(np.rad2deg(e[:, 1].max())))


def write_new(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, indent=2, allow_nan=False)+'\n'
    with path.open('x') as f: f.write(text)
