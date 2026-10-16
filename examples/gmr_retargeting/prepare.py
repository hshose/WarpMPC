#!/usr/bin/env python3
"""Prepare joint targets from AMASS SMPL-X (ACCAD) or SMPL-H (BMLhandball).

Uses the licensed body's shape blend shapes and joint regressor, then exact
skeletal forward kinematics. No mesh skinning, Torch, GUI or approximate human
skeleton is needed. The first 22 joints are shared by SMPL-X and SMPL-H.
"""

from __future__ import annotations

import argparse
import functools
import io
import json
import pathlib
import pickle
import tarfile

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

NAMES = [
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
]


def motion_members(path):
    """Stream selected archives without extracting arbitrary archive paths."""
    if path.is_dir():
        for member in sorted(path.rglob("*.npz")):
            yield str(member.relative_to(path)), member.read_bytes()
    elif path.name.endswith(".npz"):
        yield path.name, path.read_bytes()
    else:
        with tarfile.open(path, "r|*") as archive:
            for member in archive:
                if (
                    member.isfile()
                    and member.name.endswith(".npz")
                    and "stagei.npz" not in member.name
                ):
                    yield member.name, archive.extractfile(member).read()


def find_body_model(root, kind, gender):
    basename = f"{kind.upper()}_{gender.upper()}"
    for suffix in ("npz", "pkl"):
        filename = f"{basename}.{suffix}"
        choices = [
            root / filename,
            root / kind / filename,
            root / kind / gender / f"model.{suffix}",
            root / gender / f"model.{suffix}",
        ]
        for path in choices:
            if path.is_file():
                return path
    raise FileNotFoundError(
        f"AMASS {kind}/{gender} requires the licensed {basename}.npz or .pkl in {root} "
        f"or {root/kind}. Supply --body-models; no synthetic skeleton is substituted."
    )


@functools.lru_cache(maxsize=2)
def load_body_model(path):
    """Read trusted user-supplied NPZ or GMR's documented pickle body models."""
    if path.suffix.lower() == ".pkl":
        with path.open("rb") as stream:
            return pickle.load(stream, encoding="latin1")
    with np.load(path, allow_pickle=True) as body:
        return dict(body)


def joints_from_motion(data, model_path):
    # Body model files are trusted user-supplied assets. J_regressor may be sparse.
    body = load_body_model(model_path)
    template = np.asarray(body["v_template"], dtype=np.float64)
    shapes = np.asarray(body["shapedirs"], dtype=np.float64)
    betas = np.asarray(data["betas"]).ravel()
    if shapes.shape[-1] < len(betas):
        raise ValueError(
            f"{model_path} has {shapes.shape[-1]} shape coefficients, motion needs {len(betas)}"
        )
    regressor = body["J_regressor"]
    if regressor.dtype == object:
        regressor = regressor.item()
    rest = np.asarray(
        regressor @ (template + np.einsum("vck,k->vc", shapes[..., : len(betas)], betas))
    )[:22]
    tree = np.asarray(body["kintree_table"])
    lookup = {int(joint): i for i, joint in enumerate(tree[1])}
    parents = [-1] + [lookup[int(parent)] for parent in tree[0, 1:22]]
    pose = np.asarray(data["poses"])
    local = Rotation.from_rotvec(pose[:, :66].reshape(-1, 3)).as_matrix().reshape(-1, 22, 3, 3)
    positions = np.empty((len(pose), 22, 3))
    rotations = np.empty_like(local)
    positions[:, 0] = rest[0] + data["trans"]
    rotations[:, 0] = local[:, 0]
    for joint in range(1, 22):
        parent = parents[joint]
        if not 0 <= parent < joint:
            raise ValueError("Unexpected SMPL joint tree")
        positions[:, joint] = positions[:, parent] + np.einsum(
            "bij,j->bi", rotations[:, parent], rest[joint] - rest[parent]
        )
        rotations[:, joint] = rotations[:, parent] @ local[:, joint]
    local_quats = (
        Rotation.from_matrix(local.reshape(-1, 3, 3)).as_quat(scalar_first=True).reshape(-1, 22, 4)
    )
    return positions, local_quats, parents


def resample(positions, quats, parents, source_fps, target_fps):
    # Match GMR's offline linspace, linear positions and local SLERP, followed
    # by composition along the joint tree to obtain global orientations.
    skip = max(1, int(source_fps / target_fps))
    count = len(positions) // skip
    if count < 2:
        raise ValueError("Motion too short to resample")
    old = np.arange(len(positions))
    new = np.linspace(0, len(positions) - 1, count)
    pos = np.stack(
        [np.interp(new, old, positions[:, j, k]) for j in range(22) for k in range(3)], axis=1
    ).reshape(count, 22, 3)
    global_rotations = []
    for j in range(22):
        local = Slerp(old, Rotation.from_quat(quats[:, j], scalar_first=True))(new)
        global_rotations.append(local if j == 0 else global_rotations[parents[j]] * local)
    rot = np.stack([rotation.as_quat(scalar_first=True) for rotation in global_rotations], axis=1)
    return pos, rot


def load_frames(path, max_frames=None):
    with np.load(path, allow_pickle=False) as data:
        pos, quat = data["positions"], data["quaternions"]
        names = data["names"].tolist()
        starts = set(data["starts"].tolist())
        height = float(data["human_height"])
        info = json.loads(str(data["metadata"]))
    count = len(pos) if max_frames is None else min(max_frames, len(pos))
    if count < 1:
        raise ValueError("No input frames")
    frames = [
        {name: (pos[i, j].copy(), quat[i, j].copy()) for j, name in enumerate(names)}
        for i in range(count)
    ]
    return frames, {s for s in starts if s < count}, height, info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--amass", type=pathlib.Path, required=True, help="Archive, directory or single motion NPZ"
    )
    parser.add_argument("--body-models", type=pathlib.Path, required=True)
    parser.add_argument(
        "--model-type", choices=["smplx", "smplh"], help="Otherwise infer from pose dimension"
    )
    parser.add_argument("--clips", type=int, default=1)
    parser.add_argument("--frames-per-clip", type=int, default=128)
    parser.add_argument("--fps", type=float, default=30)
    parser.add_argument("--contains", default="", help="Select archive members by substring")
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    if args.clips < 1 or args.frames_per_clip < 2 or args.fps <= 0:
        parser.error("clips>=1, frames-per-clip>=2 and fps>0 required")
    positions, quaternions, starts, clips = [], [], [], []
    height = None
    for name, content in motion_members(args.amass):
        if args.contains not in name:
            continue
        with np.load(io.BytesIO(content), allow_pickle=False) as data:
            if "poses" not in data or len(data["poses"]) < 2:
                continue
            dimension = data["poses"].shape[1]
            kind = args.model_type or {165: "smplx", 156: "smplh"}.get(dimension)
            if kind is None:
                raise ValueError(f"Unknown pose format {dimension} in {name}")
            gender = data["gender"].item()
            gender = gender.decode() if isinstance(gender, bytes) else str(gender)
            body_path = find_body_model(args.body_models, kind, gender)
            source_fps = float(
                data["mocap_frame_rate" if "mocap_frame_rate" in data else "mocap_framerate"]
            )
            human_height = float(1.66 + 0.1 * np.asarray(data["betas"]).ravel()[0])
            # A prepared file has a single morphology so GMR's scaling is identical.
            if height is not None and not np.isclose(height, human_height):
                continue
            height = human_height
            p, q, parents = joints_from_motion(data, body_path)
            p, q = resample(p, q, parents, source_fps, args.fps)
        p, q = p[: args.frames_per_clip], q[: args.frames_per_clip]
        starts.append(sum(len(x) for x in positions))
        positions.append(p)
        quaternions.append(q)
        clips.append(
            {
                "member": name,
                "frames": len(p),
                "source_fps": source_fps,
                "model_type": kind,
                "body_model": str(body_path.resolve()),
                "gender": gender,
            }
        )
        print(f"Prepared {name}: {kind}, {len(p)} frames", flush=True)
        if len(clips) == args.clips:
            break
    if len(clips) != args.clips:
        raise ValueError(
            f"Only found {len(clips)} matching clips with the same morphology; wanted {args.clips}"
        )
    metadata = {
        "dataset": args.amass.name,
        "amass": True,
        "clips": clips,
        "target_fps": args.fps,
        "preprocessing": "Exact shape-dependent skeletal FK; GMR linear global positions, local SLERP then parent composition",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        positions=np.concatenate(positions),
        quaternions=np.concatenate(quaternions),
        names=np.array(NAMES),
        starts=np.array(starts),
        human_height=height,
        metadata=json.dumps(metadata),
    )


if __name__ == "__main__":
    main()
