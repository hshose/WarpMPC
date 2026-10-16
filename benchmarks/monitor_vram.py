#!/usr/bin/env python3
"""Run a benchmark unchanged and sample NVML resident memory in a separate process.

Sampled peaks include allocator reservations and caches; no JAX/CUDA APIs are
called in the benchmark, and short-lived allocations between polls may be missed.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def ancestors(pid):
    seen = set()
    while pid > 1 and pid not in seen:
        seen.add(pid)
        try:
            text = Path(f'/proc/{pid}/stat').read_text()
            pid = int(text[text.rfind(')')+2:].split()[1])
        except (OSError, ValueError, IndexError): break
    return seen


def choose_case(pid, active, root_pid):
    chain = ancestors(pid)
    if root_pid not in chain: return None
    # Walk upwards to select the closest explicitly labeled owner.
    current = pid
    for _ in range(len(chain)):
        if current in active: return active[current]
        try:
            text=Path(f'/proc/{current}/stat').read_text()
            current=int(text[text.rfind(')')+2:].split()[1])
        except (OSError, ValueError, IndexError): break
    return dict(case_id='unlabeled', labels={})


def summarize(out):
    events = {}
    for path in (out/'cases').glob('*.jsonl'):
        for line in path.read_text().splitlines():
            e=json.loads(line); row=events.setdefault(e['case_id'], dict(case_id=e['case_id'],pid=e['pid'],labels=e['labels']))
            row['start_ns' if e['event']=='start' else 'end_ns']=e['time_ns']
            row['boundary_status']=e['event']
    peaks={}
    with (out/'samples.csv').open() as f:
        for row in csv.DictReader(f):
            event = events.get(row['case_id'])
            if event and not (event.get('start_ns', 0) <= int(row['time_ns']) <= event.get('end_ns', 2**63-1)):
                continue
            key=(row['case_id'],row['gpu_uuid'])
            p=peaks.setdefault(key,dict(peak_bytes=0,samples=0))
            p['peak_bytes']=max(p['peak_bytes'],int(row['used_bytes']));p['samples']+=1
    rows=[]
    for cid,e in events.items():
        matches=[(gpu,v) for (case,gpu),v in peaks.items() if case==cid]
        if not matches: matches=[('',dict(peak_bytes=None,samples=0))]
        for gpu,v in matches:
            rows.append(dict(case_id=cid,gpu_uuid=gpu,**v,peak_mib=v['peak_bytes']/2**20 if v['peak_bytes'] is not None else None,
                             boundary_status=e['boundary_status'],start_ns=e.get('start_ns'),end_ns=e.get('end_ns'),
                             **e['labels']))
    if rows:
        fields=list(dict.fromkeys(k for row in rows for k in row))
        with (out/'case_peaks.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--interval',type=float,default=1.0)
    parser.add_argument('--summarize-only',action='store_true')
    parser.add_argument('command',nargs=argparse.REMAINDER)
    args=parser.parse_args();out=args.output_dir.resolve()
    if args.summarize_only: summarize(out);return
    if args.interval<=0: parser.error('interval must be positive')
    command=args.command[1:] if args.command[:1]==['--'] else args.command
    if not command: parser.error('benchmark command required')
    import pynvml as nv
    nv.nvmlInit()
    handles=[nv.nvmlDeviceGetHandleByIndex(i) for i in range(nv.nvmlDeviceGetCount())]
    out.mkdir(parents=True,exist_ok=True);(out/'cases').mkdir(exist_ok=True)
    env=dict(os.environ,VRAM_CASE_DIR=str(out/'cases'))
    metadata=dict(command=command,interval_seconds=args.interval,measurement='NVML per-process resident GPU bytes, including allocator reservations and cached blocks',
                  limitations='Sampled lower bound on peak; short allocations may be missed. Not live tensor bytes. No allocator/timing/solver changes.',
                  job_id=os.environ.get('SLURM_JOB_ID'),start_ns=time.time_ns(),status='running',errors=[])
    meta_path=out/'metadata.json';meta_path.write_text(json.dumps(metadata,indent=2))
    process=subprocess.Popen(command,env=env)
    # Lower only the sampler's CPU priority; the benchmark was spawned first.
    os.nice(10)
    def forward_signal(sig, frame):
        if process.poll() is None: process.send_signal(sig)
    signal.signal(signal.SIGTERM,forward_signal);signal.signal(signal.SIGINT,forward_signal)
    with (out/'samples.csv').open('w',buffering=1) as stream:
        writer=csv.DictWriter(stream,fieldnames=['time_ns','case_id','gpu_uuid','used_bytes','pids']);writer.writeheader()
        while process.poll() is None:
            tick=time.monotonic();active={}
            for path in (out/'cases').glob('*.json'):
                try: record=json.loads(path.read_text());active[record['pid']]=record
                except (OSError,ValueError): continue
            totals={}
            for handle in handles:
                gpu=nv.nvmlDeviceGetUUID(handle)
                try: entries=nv.nvmlDeviceGetComputeRunningProcesses(handle)
                except nv.NVMLError as exc:
                    error=str(exc)
                    if error not in metadata['errors']: metadata['errors'].append(error)
                    continue
                for entry in entries:
                    case=choose_case(entry.pid,active,process.pid)
                    if case is None: continue
                    memory=entry.usedGpuMemory
                    if memory is None or int(memory) >= 2**63: continue
                    record=totals.setdefault((case['case_id'],gpu),dict(used_bytes=0,pids=[],owner_pid=case.get('pid')))
                    record['used_bytes']+=int(memory);record['pids'].append(entry.pid)
            now=time.time_ns()
            for (cid,gpu),record in totals.items():
                # Discard samples straddling a case transition rather than
                # attribute a new batch's allocations to the previous batch.
                if record['owner_pid'] is not None:
                    try:
                        current=json.loads((out/'cases'/f"{record['owner_pid']}.json").read_text())
                    except (OSError, ValueError):
                        continue
                    if current['case_id'] != cid: continue
                writer.writerow(dict(time_ns=now,case_id=cid,gpu_uuid=gpu,used_bytes=record['used_bytes'],pids=json.dumps(record['pids'])))
            try: process.wait(timeout=max(.001,args.interval-(time.monotonic()-tick)))
            except subprocess.TimeoutExpired: pass
    code=process.wait();metadata.update(status='complete',returncode=code,end_ns=time.time_ns())
    meta_path.write_text(json.dumps(metadata,indent=2));summarize(out);nv.nvmlShutdown()
    raise SystemExit(code if code>=0 else 128-code)

if __name__=='__main__': main()
