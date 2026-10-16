#!/usr/bin/env python3
"""Check prepared AMASS targets against SMPL-X and GMR's original resampler.

Optional CPU-only check: requires torch and smplx in addition to the example
dependencies. No body models or motion data are bundled with the example.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import pathlib
import sys
import types

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.spatial.transform import Rotation

from examples.gmr_retargeting.prepare import NAMES, find_body_model, motion_members


def original_resampler(gmr_root):
    # Avoid importing GMR's package initializer, which also loads its viewer.
    directory = gmr_root.resolve() / "general_motion_retargeting"
    package = types.ModuleType("general_motion_retargeting")
    package.__path__ = [str(directory)]
    sys.modules[package.__name__] = package
    path = directory / "utils/smpl.py"
    spec = importlib.util.spec_from_file_location("general_motion_retargeting.utils.smpl", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.get_smplx_data_offline_fast


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--amass", type=pathlib.Path, required=True)
    parser.add_argument("--data", type=pathlib.Path, required=True)
    parser.add_argument("--body-models", type=pathlib.Path, required=True)
    parser.add_argument("--gmr-root", type=pathlib.Path, default=ROOT / "resources/GMR")
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument(
        "--batch-size", type=int, default=128, help="Bound CPU SMPL-X mesh memory for long clips"
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    import smplx
    import torch

    torch.set_num_threads(1)
    resampler = original_resampler(args.gmr_root)
    with np.load(args.data, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"]))
        np.testing.assert_array_equal(data["names"], NAMES)
        expected_pos, expected_quat = data["positions"], data["quaternions"]
        clips = {
            clip["member"]: (clip, int(start))
            for clip, start in zip(metadata["clips"], data["starts"])
        }
    records, models, hashes = [], {}, {}
    for name, content in motion_members(args.amass):
        if name not in clips:
            continue
        clip, start = clips[name]
        if clip["model_type"] != "smplx":
            raise ValueError("The original GMR SMPL-X preprocessing check requires SMPL-X input")
        with np.load(io.BytesIO(content), allow_pickle=False) as motion:
            gender = clip["gender"]
            betas = np.asarray(motion["betas"]).reshape(1, -1)
            key = gender, betas.shape[1]
            if key not in models:
                path = find_body_model(args.body_models, "smplx", gender)
                # AMASS supplies 16 coefficients; smplx otherwise defaults to 10.
                models[key] = smplx.create(
                    str(path), "smplx", gender=gender, num_betas=betas.shape[1], use_pca=False
                )
                hashes[str(path.resolve())] = sha256(path)
            body = models[key]
            frames = len(motion["poses"])
            parts = {name: [] for name in ("global_orient", "full_pose", "joints")}
            with torch.no_grad():
                for first in range(0, frames, args.batch_size):
                    end = min(frames, first + args.batch_size)
                    count = end - first
                    result = body(
                        betas=torch.as_tensor(betas, dtype=torch.float32),
                        global_orient=torch.as_tensor(
                            motion["root_orient"][first:end], dtype=torch.float32
                        ),
                        body_pose=torch.as_tensor(
                            motion["pose_body"][first:end], dtype=torch.float32
                        ),
                        transl=torch.as_tensor(motion["trans"][first:end], dtype=torch.float32),
                        left_hand_pose=torch.zeros(count, 45),
                        right_hand_pose=torch.zeros(count, 45),
                        jaw_pose=torch.zeros(count, 3),
                        leye_pose=torch.zeros(count, 3),
                        reye_pose=torch.zeros(count, 3),
                        expression=torch.zeros(count, 10),
                        return_full_pose=True,
                        return_verts=False,
                    )
                    for field in parts:
                        parts[field].append(getattr(result, field))
            result = types.SimpleNamespace(
                **{field: torch.cat(values) for field, values in parts.items()}
            )
            reference, _ = resampler(motion, body, result, tgt_fps=metadata["target_fps"])
        count = clip["frames"]
        positions = np.array([[frame[joint][0] for joint in NAMES] for frame in reference[:count]])
        quaternions = np.array(
            [[frame[joint][1] for joint in NAMES] for frame in reference[:count]]
        )
        error_pos = float(np.max(np.abs(positions - expected_pos[start : start + count])))
        ref_rot = Rotation.from_quat(quaternions.reshape(-1, 4), scalar_first=True)
        own_rot = Rotation.from_quat(
            expected_quat[start : start + count].reshape(-1, 4), scalar_first=True
        )
        error_rot = float(np.max((ref_rot.inv() * own_rot).magnitude()))
        record = dict(
            member=name,
            frames=count,
            max_position_error_m=error_pos,
            max_orientation_error_rad=error_rot,
            num_betas=key[1],
        )
        records.append(record)
        print(json.dumps(record), flush=True)
        if max(error_pos, error_rot) > 2e-5:
            raise AssertionError(f"AMASS preprocessing differs from SMPL-X/GMR: {record}")
        if len(records) == len(clips):
            break
    if len(records) != len(clips):
        raise ValueError("Some prepared motion clips were not found in the source archive")
    report = dict(
        status="ok",
        clips=records,
        body_model_sha256=hashes,
        input_sha256=sha256(args.amass),
        prepared_data_sha256=sha256(args.data),
        preprocessing_sha256=sha256(pathlib.Path(__file__).with_name("prepare.py")),
        gmr_resampler_sha256=sha256(args.gmr_root / "general_motion_retargeting/utils/smpl.py"),
        versions={name: importlib.metadata.version(name) for name in ("torch", "smplx", "numpy")},
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
