"""Local coordinator for the authorized RTX run and live-source training audit.

The remote qualification must already be running. No robot application is
executed: the last stage only builds a new qualified hardware profile.
"""
import argparse,os,shlex,subprocess,sys
from pathlib import Path


def run(a):
    q=shlex.quote;remote=a.remote_root;result=remote/'results'/a.run_name
    sdk=a.sdk_source;reference=remote/'results/juggling_reference';observer=remote/'results/observer_build/libjuggling_batch_observer.so'
    audit=result/'pretraining_audit';ssh=['ssh','-F',str(a.ssh_config),'-o','ServerAliveInterval=30',a.host]
    env=dict(PYTHON=a.remote_python,ROOT=str(remote),RESULTS_DIR=str(result),REFERENCE=str(reference),OBSERVER=str(observer),
             FRANKA_SOURCE=str(remote/'live_source'),CALIBRATION=str(remote/'live_source/runs/20261006_sim_quality/model.json'),
             EIGEN_DIR=str(sdk/'build/eigen_install/include/eigen3'),CMAKE=str(sdk/'.venv/bin/cmake'),
             PYTHONPATH=str(remote/'external'),WARP_CACHE_PATH='/tmp/franka-warp',JAX_COMPILATION_CACHE_DIR=str(remote/'results/jax_cache'),
             LD_LIBRARY_PATH=str(sdk/'.venv/lib/python3.13/site-packages/casadi'),XLA_PYTHON_CLIENT_PREALLOCATE='false',
             OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4',COLLECT_BATCH_SIZE='60000',COLLECT_MICROBATCH_SIZE=str(a.microbatch),
             TRAIN_AUDIT_DIR=str(audit),BUILD_JOBS='8')
    script=['#!/usr/bin/env bash','set -euo pipefail','cd '+q(str(remote)),'mkdir -p '+q(str(result))]
    for key,value in env.items():script.append('export '+key+'='+q(value))
    for name,target in [('native',a.native_qualification),('warp',a.warp_qualification)]:
        dest=result/('qualification_'+name)
        script.append('if [[ ! -e '+q(str(dest))+' ]]; then ln -s '+q(str(target))+' '+q(str(dest))+'; fi')
    script+=['deadline=$((SECONDS+21600))','while [[ ! -f '+q(str(a.warp_qualification/'rollout.json'))+' ]]; do',
             '  if ! kill -0 '+str(a.qualification_pid)+' 2>/dev/null; then echo "Qualification process stopped without a report" >&2; exit 1; fi',
             '  if ((SECONDS>deadline)); then echo "Qualification timeout" >&2; exit 1; fi','  sleep 30','done',
             'bash examples/juggling_ampc/run_disk_ampc_legacy.sh',
             'export FRANKA_SOURCE='+q(str(sdk))+' BASE_PROFILE='+q(str(a.base_profile))+' FRANKA_PYTHON='+q(a.remote_python),
             'bash examples/juggling_ampc/build_hardware.sh '+q(str(result/'policy/eigen'))+' '+q(str(result/'hardware')),
             'echo '+q('All gates completed. Hardware profile built; robot not executed.'),
             'echo '+q('FRANKA_SOURCE='+str(sdk)+' bash '+str(remote/'examples/juggling_ampc/run_hardware.sh')+' '+str(result/'hardware'))]
    remote_script=result/'run_pipeline.sh'
    write="import pathlib,sys; p=pathlib.Path(sys.argv[1]); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(sys.stdin.read())"
    subprocess.run(ssh+[shlex.join([a.remote_python,'-c',write,str(remote_script)])],input='\n'.join(script)+'\n',text=True,check=True)
    a.logs.mkdir(parents=True,exist_ok=True)
    with (a.logs/'pipeline.log').open('a') as pipeline_log,(a.logs/'audit.log').open('a') as audit_log:
        command='bash '+q(str(remote_script))+' > '+q(str(result/'pipeline.log'))+' 2>&1'
        job=subprocess.Popen(ssh+[command],stdout=pipeline_log,stderr=subprocess.STDOUT)
        watcher=subprocess.Popen([sys.executable,str(Path(__file__).with_name('audit_remote_training.py')),
            '--host',a.host,'--ssh-config',str(a.ssh_config),'--source',str(a.source),'--calibration',str(a.calibration),
            '--remote-directory',str(audit),'--remote-python',a.remote_python,'--timeout','259200','--poll-seconds','60'],stdout=audit_log,stderr=subprocess.STDOUT)
        try:
            print('Remote pipeline: '+str(result/'pipeline.log'),flush=True)
            print('Local authoritative audit: '+str(a.logs/'audit.log'),flush=True)
            code=job.wait()
            if code:raise RuntimeError(f'Remote pipeline stopped with status {code}; inspect its stage log')
        finally:
            if watcher.poll() is None:watcher.terminate();watcher.wait()
    print('Qualified hardware profile: '+str(result/'hardware'),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--host',required=True);p.add_argument('--ssh-config',type=Path,default=Path.home() / '.ssh/config')
    p.add_argument('--remote-root',type=Path,required=True);p.add_argument('--sdk-source',type=Path,required=True);p.add_argument('--base-profile',type=Path,required=True)
    p.add_argument('--native-qualification',type=Path,required=True);p.add_argument('--warp-qualification',type=Path,required=True);p.add_argument('--qualification-pid',type=int,required=True)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--calibration',type=Path,required=True);p.add_argument('--logs',type=Path,required=True)
    p.add_argument('--remote-python',default='/tmp/juggling-newton-venv/bin/python');p.add_argument('--run-name',default='rtx_ampc_20261006');p.add_argument('--microbatch',type=int,default=8192)
    a=p.parse_args()
    if a.microbatch<100 or a.qualification_pid<1 or '/' in a.run_name:raise ValueError('Invalid collection or run parameters')
    run(a)
