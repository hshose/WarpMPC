#!/usr/bin/env python3
"""Build a subject-balanced bank of nonoverlapping, real AMASS chunks.

Subject height is folded into the input positions, before GMR's fixed offsets.
This is algebraically equivalent to each subject's original GMR scale table;
the original implementation verifies target equivalence for every chunk.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from collections import defaultdict, deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from examples.gmr_retargeting.prepare import (
    NAMES,
    find_body_model,
    joints_from_motion,
    motion_members,
    resample,
)


def select_chunks(inventories, length, limit, seed):
    subjects = defaultdict(list)
    for inventory in inventories:
        for clip in inventory["clips"]:
            for start in range(0, clip["frames"] - length + 1, length):
                subjects[clip["subject"]].append(clip | {"first_frame": start, "frames": length})
    rng = np.random.default_rng(seed)
    queues = []
    for subject in sorted(subjects):
        candidates = subjects[subject]
        rng.shuffle(candidates)
        queues.append(deque(candidates))
    rng.shuffle(queues)
    selected = []
    while queues and len(selected) < limit:
        remaining = []
        for queue in queues:
            if len(selected) < limit:
                selected.append(queue.popleft())
            if queue:
                remaining.append(queue)
        queues = remaining
    return selected


def fold_height(positions, height, reference_height):
    return positions * (height / reference_height)


def target_vector(gmr):
    return np.concatenate(
        [
            np.r_[
                task.transform_target_to_world.translation(),
                task.transform_target_to_world.rotation().wxyz,
            ]
            for tasks in (gmr.tasks1, gmr.tasks2)
            for task in tasks
        ]
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inventory", type=Path, nargs="+", required=True)
    p.add_argument("--body-models", type=Path, required=True)
    p.add_argument("--gmr-root", type=Path, default=ROOT / "resources/GMR")
    p.add_argument("--chunk-size", type=int, default=2048)
    p.add_argument("--max-chunks", type=int, default=128)
    p.add_argument("--seed", type=int, default=20260930)
    p.add_argument("--reference-height", type=float, default=1.66)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--selection-only", action="store_true")
    args = p.parse_args()
    if args.chunk_size < 2 or args.max_chunks < 1 or args.reference_height <= 0:
        p.error("Positive height/chunk count and chunk size >=2 required")
    inventories = [json.loads(path.read_text()) for path in args.inventory]
    rates = {inv["summary"]["target_fps"] for inv in inventories}
    if len(rates) != 1:
        p.error("Inventory frame rates must match")
    fps = rates.pop()
    selected = select_chunks(inventories, args.chunk_size, args.max_chunks, args.seed)
    if not selected:
        p.error("No complete chunks available")
    summary = dict(
        chunks=len(selected),
        frames=len(selected) * args.chunk_size,
        subjects=len({c["subject"] for c in selected}),
        sessions=len({c.get("session", c["subject"]) for c in selected}),
        clips=len({(c["archive"], c["member"]) for c in selected}),
        datasets=sorted({Path(c["archive"]).name for c in selected}),
        chunk_size=args.chunk_size,
        target_fps=fps,
        seed=args.seed,
    )
    print(json.dumps(summary), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".selection.json").write_text(
        json.dumps(dict(summary=summary, chunks=selected), indent=2) + "\n"
    )
    if args.selection_only:
        return
    from examples.gmr_retargeting.model import load_gmr

    common = load_gmr(args.gmr_root, args.reference_height)
    wanted = defaultdict(lambda: defaultdict(list))
    for i, clip in enumerate(selected):
        wanted[clip["archive"]][clip["member"]].append(i)
    positions, quaternions, records = (
        [None] * len(selected),
        [None] * len(selected),
        [None] * len(selected),
    )
    max_target_error = 0.0
    body_hashes = {}
    for archive, members in wanted.items():
        for member, content in motion_members(Path(archive)):
            if member not in members:
                continue
            ids = members.pop(member)
            clip = selected[ids[0]]
            if hashlib.sha256(content).hexdigest() != clip["source_sha256"]:
                raise ValueError(f"Source changed after inventory: {member}")
            with np.load(io.BytesIO(content), allow_pickle=False) as data:
                kind = {165: "smplx", 156: "smplh"}[data["poses"].shape[1]]
                body_path = find_body_model(args.body_models, kind, clip["gender"])
                if str(body_path) not in body_hashes:
                    with body_path.open("rb") as stream:
                        body_hashes[str(body_path)] = hashlib.file_digest(
                            stream, "sha256"
                        ).hexdigest()
                pos, quat, parents = joints_from_motion(data, body_path)
                pos, quat = resample(pos, quat, parents, clip["source_fps"], fps)
            original = load_gmr(args.gmr_root, clip["height"])
            for i in ids:
                record = selected[i]
                start, end = record["first_frame"], record["first_frame"] + args.chunk_size
                raw, rotations = pos[start:end], quat[start:end]
                folded = fold_height(raw, clip["height"], args.reference_height)
                for t in (0, args.chunk_size // 2, args.chunk_size - 1):
                    original.update_targets(
                        {
                            name: (raw[t, j].copy(), rotations[t, j].copy())
                            for j, name in enumerate(NAMES)
                        }
                    )
                    common.update_targets(
                        {
                            name: (folded[t, j].copy(), rotations[t, j].copy())
                            for j, name in enumerate(NAMES)
                        }
                    )
                    error = float(np.max(np.abs(target_vector(original) - target_vector(common))))
                    max_target_error = max(max_target_error, error)
                    if error > 1e-12:
                        raise AssertionError(f"Subject height folding changed GMR targets: {error}")
                positions[i], quaternions[i] = folded, rotations.copy()
                records[i] = record | dict(
                    model_type=kind,
                    body_model=str(body_path.resolve()),
                    position_scale=clip["height"] / args.reference_height,
                )
            print(f"Prepared {member}: {len(ids)} nonoverlapping chunks", flush=True)
            if not members:
                break
        if members:
            raise ValueError(f"Missing selected clips: {list(members)}")
    metadata = dict(
        dataset="AMASS diverse long chunks",
        amass=True,
        clips=records,
        target_fps=fps,
        pool=summary,
        body_model_sha256=body_hashes,
        height_folding_max_target_error=max_target_error,
        preprocessing="Exact shape-dependent skeletal FK; GMR resampling; subject height folded into raw positions before fixed GMR offsets",
        selection="Seeded round-robin across subjects, nonoverlapping within source clips; independent cold starts",
    )
    np.savez_compressed(
        args.output,
        positions=np.concatenate(positions),
        quaternions=np.concatenate(quaternions),
        names=np.array(NAMES),
        starts=np.arange(len(selected)) * args.chunk_size,
        human_height=args.reference_height,
        metadata=json.dumps(metadata),
    )
    print(
        f"Saved {args.output}; original subject-specific GMR target error {max_target_error:.3g}",
        flush=True,
    )


if __name__ == "__main__":
    main()
