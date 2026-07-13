from __future__ import annotations

import argparse
import glob
import json
import os
import pathlib
import select
import subprocess
import termios
import time
import tty

from benchmarks.embedded_ampc.config import (
    ACTIVATIONS,
    BACKENDS,
    NETWORK_SIZES,
    PRECISIONS,
    REPO_ROOT,
    TARGETS,
    BenchmarkCase,
    iter_cases,
)
from benchmarks.embedded_ampc.generate import DEFAULT_OUTPUT_ROOT, generate_case


DEFAULT_RESULTS_ROOT = REPO_ROOT / "results"


def _run_checked(stage: str, cmd: list[str], project_dir: pathlib.Path) -> None:
    try:
        subprocess.run(cmd, cwd=project_dir, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        exc.args = (*exc.args, stage)
        raise


def build_project(project_dir: pathlib.Path) -> None:
    _run_checked("lbuild", ["lbuild", "build"], project_dir)
    _run_checked("build", ["scons", "-Q", "-j16", "build"], project_dir)


def program_project(project_dir: pathlib.Path) -> None:
    _run_checked("program", ["scons", "-Q", "program"], project_dir)


def find_serial_port() -> str:
    patterns = ("/dev/serial/by-id/*STLink*", "/dev/serial/by-id/*ST-Link*", "/dev/serial/by-id/*", "/dev/ttyACM*", "/dev/ttyUSB*")
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            return os.path.realpath(matches[0])
    raise RuntimeError("could not find a serial port; pass --serial-port explicitly")


class SerialCaptureTimeout(TimeoutError):
    def __init__(self, port: str, buffer: str) -> None:
        super().__init__(f"timed out waiting for BENCH_DONE on {port}")
        self.port = port
        self.buffer = buffer


def capture_serial_result(port: str, *, baud: int = 115200, timeout_s: float | None = None) -> tuple[dict[str, object], str]:
    fd = os.open(port, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
    old_attrs = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        attrs = termios.tcgetattr(fd)
        if baud == 115200:
            attrs[4] = termios.B115200
            attrs[5] = termios.B115200
        attrs[2] |= termios.CLOCAL | termios.CREAD
        if hasattr(termios, "CRTSCTS"):
            attrs[2] &= ~termios.CRTSCTS
        termios.tcsetattr(fd, termios.TCSANOW, attrs)
        termios.tcflush(fd, termios.TCIFLUSH)

        deadline = None if timeout_s is None or timeout_s <= 0.0 else time.monotonic() + timeout_s
        buffer = ""
        pending = ""
        result: dict[str, object] | None = None
        while deadline is None or time.monotonic() < deadline:
            ready, _, _ = select.select([fd], [], [], 0.2)
            if not ready:
                continue
            chunk = os.read(fd, 4096).decode("utf-8", errors="replace")
            buffer += chunk
            pending += chunk
            while "\n" in pending:
                line, pending = pending.split("\n", 1)
                line = line.rstrip("\r")
                if "BENCH_RESULT" in line:
                    payload = line.split("BENCH_RESULT", 1)[1].strip()
                    result = json.loads(payload)
                if "BENCH_DONE" in line and result is not None:
                    return result, buffer
        raise SerialCaptureTimeout(port, buffer)
    finally:
        termios.tcsetattr(fd, termios.TCSANOW, old_attrs)
        os.close(fd)


def run_cases(
    cases: list[BenchmarkCase],
    *,
    output_root: pathlib.Path,
    results_dir: pathlib.Path | None,
    serial_port: str | None,
    build_only: bool,
    skip_program: bool,
    timeout_s: float | None,
) -> pathlib.Path:
    if results_dir is None:
        results_dir = DEFAULT_RESULTS_ROOT / f"embedded_ampc_{cases[0].target}_{time.strftime('%Y%m%d_%H%M%S')}"
    results_dir.mkdir(parents=True, exist_ok=True)
    stamp = f"{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
    output_jsonl = results_dir / f"embedded_ampc_{cases[0].target}_{stamp}.jsonl"
    port = serial_port or (None if skip_program or build_only else find_serial_port())
    with output_jsonl.open("a", encoding="utf-8") as fp:
        for index, case in enumerate(cases, start=1):
            print(f"[{index}/{len(cases)}] {case.case_id}", flush=True)
            case_start = time.monotonic()
            project_dir: pathlib.Path | None = None
            try:
                project_dir = generate_case(case, output_root=output_root, host_check=True)
                if skip_program:
                    result = _status_row(case, project_dir, "host_checked")
                else:
                    build_project(project_dir)
                    if build_only:
                        result = _status_row(case, project_dir, "built")
                    else:
                        program_project(project_dir)
                        assert port is not None
                        result, raw_log = capture_serial_result(port, timeout_s=timeout_s)
                        result["project_dir"] = str(project_dir)
                        result["raw_log"] = raw_log
            except Exception as exc:  # Keep a long sweep moving after fit/build/serial failures.
                result = _failure_row(case, project_dir, exc)
            elapsed_s = time.monotonic() - case_start
            result["elapsed_s"] = elapsed_s
            fp.write(json.dumps(result, sort_keys=True) + "\n")
            fp.flush()
            print(f"    -> {result.get('status', 'ok')} in {elapsed_s:.1f}s", flush=True)
    return output_jsonl


def _status_row(case: BenchmarkCase, project_dir: pathlib.Path, status: str) -> dict[str, object]:
    return {
        "case_id": case.case_id,
        "target": case.target,
        "backend": case.backend,
        "precision": case.precision,
        "activation": case.activation,
        "hidden_layers": case.hidden_layers,
        "width": case.width,
        "section": case.section,
        "project_dir": str(project_dir),
        "status": status,
    }


def _failure_row(case: BenchmarkCase, project_dir: pathlib.Path | None, exc: Exception) -> dict[str, object]:
    row: dict[str, object] = {
        "case_id": case.case_id,
        "target": case.target,
        "backend": case.backend,
        "precision": case.precision,
        "activation": case.activation,
        "hidden_layers": case.hidden_layers,
        "width": case.width,
        "section": case.section,
        "project_dir": str(project_dir) if project_dir is not None else None,
        "status": "failed",
        "error_type": type(exc).__name__,
        "error": str(exc),
    }
    if isinstance(exc, subprocess.CalledProcessError):
        stage = exc.args[-1] if exc.args and isinstance(exc.args[-1], str) else "command"
        row.update(
            {
                "status": f"{stage}_failed",
                "returncode": exc.returncode,
                "stdout_tail": (exc.stdout or "")[-4000:],
                "stderr_tail": (exc.stderr or "")[-4000:],
            }
        )
    if isinstance(exc, SerialCaptureTimeout):
        row["raw_log_tail"] = exc.buffer[-4000:]
    return row


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=sorted(TARGETS), required=True)
    parser.add_argument("--hidden-layers", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--activation", action="append")
    parser.add_argument("--backend", action="append")
    parser.add_argument("--precision", action="append")
    parser.add_argument("--section", action="append")
    parser.add_argument("--case-id", action="append")
    parser.add_argument("--output-root", type=pathlib.Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--results-dir", type=pathlib.Path)
    parser.add_argument("--serial-port")
    parser.add_argument("--timeout-s", type=float)
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--skip-program", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--order", choices=("default", "balanced"), default="default")
    parser.add_argument("--list", action="store_true")
    return parser.parse_args()


def _expand_values(values: list[str] | None, *, valid: tuple[str, ...] | None, name: str) -> list[str | None]:
    if not values:
        return [None]
    expanded: list[str] = []
    for value in values:
        expanded.extend(item.strip() for item in value.split(",") if item.strip())
    if valid is not None:
        invalid = [value for value in expanded if value not in valid]
        if invalid:
            raise SystemExit(f"invalid {name}: {', '.join(invalid)}; expected one of {', '.join(valid)}")
    unique: list[str] = []
    for value in expanded:
        if value not in unique:
            unique.append(value)
    return unique


def _collect_cases(args: argparse.Namespace) -> list[BenchmarkCase]:
    cases: list[BenchmarkCase] = []
    activations = _expand_values(args.activation, valid=ACTIVATIONS, name="activation")
    backends = _expand_values(args.backend, valid=BACKENDS, name="backend")
    precisions = _expand_values(args.precision, valid=PRECISIONS, name="precision")
    sections = _expand_values(args.section, valid=None, name="section")

    for activation in activations:
        for backend in backends:
            for precision in precisions:
                for section in sections:
                    cases.extend(
                        iter_cases(
                            target=args.target,
                            hidden_layers=args.hidden_layers,
                            width=args.width,
                            activation=activation,
                            backend=backend,
                            precision=precision,
                            section=section,
                        )
                    )
    if args.case_id:
        case_ids = set(_expand_values(args.case_id, valid=None, name="case-id"))
        cases = [case for case in cases if case.case_id in case_ids]
    if args.order == "balanced":
        cases = sorted(cases, key=_balanced_case_key)
    if args.offset:
        cases = cases[args.offset :]
    if args.limit is not None:
        cases = cases[: args.limit]
    return cases


def _balanced_case_key(case: BenchmarkCase) -> tuple[int, int, int, int, int]:
    size_index = NETWORK_SIZES.index((case.hidden_layers, case.width))
    sections = TARGETS[case.target].default_sections
    section_index = sections.index(case.section) if case.section in sections else len(sections)
    activation_order = {value: index for index, value in enumerate(("relu", "tanh", "leaky_relu", "elu"))}
    precision_order = {value: index for index, value in enumerate(PRECISIONS)}
    backend_order = {value: index for index, value in enumerate(("simple", "cmsis", "eigen"))}
    return (
        size_index,
        section_index,
        activation_order.get(case.activation, len(activation_order)),
        precision_order.get(case.precision, len(precision_order)),
        backend_order.get(case.backend, len(backend_order)),
    )


def main() -> None:
    args = _parse_args()
    cases = _collect_cases(args)
    if not cases:
        raise SystemExit("no benchmark cases selected")
    if args.list:
        for case in cases:
            print(case.case_id)
        return
    results_path = run_cases(
        cases,
        output_root=args.output_root,
        results_dir=args.results_dir,
        serial_port=args.serial_port,
        build_only=args.build_only,
        skip_program=args.skip_program,
        timeout_s=args.timeout_s,
    )
    print(results_path)


if __name__ == "__main__":
    main()
