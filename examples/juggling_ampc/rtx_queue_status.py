"""Read-only status of a detached 5/10/20 s training queue."""
import json, os, sys
from pathlib import Path
root=Path(sys.argv[1])
def alive(path):
    try:os.kill(int(path.read_text()),0);return True
    except (OSError,ValueError):return False
finished=(root/'finished.txt').exists()
print('Queue:', 'DONE' if finished else 'RUNNING' if alive(root/'queue.pid') else 'NOT RUNNING')
print('Artifacts:',root)
for seconds in (5,10,20):
    case=root/f'calibrated_{seconds}s';out=case/'artifacts';end=case/'exit_code.txt'
    if end.exists():state='DONE' if end.read_text().strip()=='0' else 'FAILED (exit '+end.read_text().strip()+')'
    elif (case/'started.txt').exists():state='RUNNING' if alive(case/'process.pid') else 'INTERRUPTED'
    else:state='WAITING'
    phases=out/'pipeline_timings.jsonl';phase='collection / compilation'
    if phases.exists():
        try:
            records=[json.loads(line) for line in phases.read_text().splitlines() if line.strip()]
            names=['collection','training','export','evaluation_student']
            last=records[-1];i=names.index(last['phase']);phase=names[i+1] if i+1<len(names) else 'complete'
        except (ValueError,KeyError,IndexError):phase='see log'
    print(f'{seconds:2d}s: {state}; phase: {phase}; checkpoint: {"yes" if (out/"policy/policy.npz").exists() else "no"}')
    comparison=out/'policy/eigen/comparison.json'
    if comparison.exists():
        r=json.loads(comparison.read_text());print(f"     mean uptime: policy {r['policy_mean_uptime_s']:.2f}s / MPC {r['expert_mean_uptime_s']:.2f}s")
    elif (out/'evaluation_student/rollout.json').exists():
        r=json.loads((out/'evaluation_student/rollout.json').read_text());print(f"     policy mean uptime: {r['mean_uptime_s']:.2f}s; 30s survival: {100*r['survival_fraction']:.1f}%")
    log=case/'run.log'
    if log.exists() and state != 'DONE':
        with log.open('rb') as stream:
            stream.seek(max(0,log.stat().st_size-4096));lines=stream.read().decode(errors='replace').splitlines()
        if lines:print('     '+lines[-1][:250])
if finished:print('All succeeded:',(root/'exit_code.txt').read_text().strip()=='0')
