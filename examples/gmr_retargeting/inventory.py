#!/usr/bin/env python3
"""Inventory real AMASS clips without extracting archives or decoding pose arrays."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from examples.gmr_retargeting.prepare import motion_members


def subject_id(member):
    session = str(Path(member).parent)
    if session.startswith("DanceDB/"):
        return "DanceDB/" + re.sub(r"^\d{8}_", "", Path(session).name)
    if session.startswith("ACCAD/"):
        match = re.match(r"(Male|Female)\d+", Path(session).name)
        if match:
            return "ACCAD/" + match.group()
    return session


def inventory(path, fps=30):
    clips = []
    for member, content in motion_members(path):
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            if "poses.npy" not in archive.namelist():
                continue
            with archive.open("poses.npy") as stream:
                version = np.lib.format.read_magic(stream)
                reader = {
                    (1, 0): np.lib.format.read_array_header_1_0,
                    (2, 0): np.lib.format.read_array_header_2_0,
                }[version]
                shape, _, _ = reader(stream)
        with np.load(io.BytesIO(content), allow_pickle=False) as data:
            rate = float(
                data["mocap_frame_rate" if "mocap_frame_rate" in data else "mocap_framerate"]
            )
            betas = np.asarray(data["betas"]).ravel()
            gender = data["gender"].item()
            gender = gender.decode() if isinstance(gender, bytes) else str(gender)
        count = shape[0] // max(1, int(rate / fps))
        clips.append(
            dict(
                archive=str(path.resolve()),
                member=member,
                subject=subject_id(member),
                session=str(Path(member).parent),
                source_frames=shape[0],
                pose_dimension=shape[1],
                source_fps=rate,
                frames=count,
                gender=gender,
                height=float(1.66 + 0.1 * betas[0]),
                betas_sha256=hashlib.sha256(betas.tobytes()).hexdigest(),
                source_sha256=hashlib.sha256(content).hexdigest(),
            )
        )
        if len(clips) % 100 == 0:
            print(f"{path.name}: {len(clips)} clips", flush=True)
    return clips


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--amass", type=Path, required=True)
    p.add_argument("--fps", type=float, default=30)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    clips = inventory(args.amass, args.fps)
    summary = dict(
        archive=str(args.amass.resolve()),
        target_fps=args.fps,
        clips=len(clips),
        subjects=len({c["subject"] for c in clips}),
        frames=sum(c["frames"] for c in clips),
        sessions=len({c["session"] for c in clips}),
        nonoverlapping_chunks={
            str(n): sum(c["frames"] // n for c in clips) for n in (64, 256, 512, 1024, 2048, 4096)
        },
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(summary=summary, clips=clips), indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
