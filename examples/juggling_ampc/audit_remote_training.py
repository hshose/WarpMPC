"""Run locally: audit the actual live source when remote preprocessing is done."""
import argparse,datetime,json,shlex,subprocess,time
from pathlib import Path
from snapshot import fingerprint
from pretraining_audit import identity


def audit(a):
    ssh=['ssh','-F',str(a.ssh_config),'-o','ConnectTimeout=15','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=2',a.host]
    request=a.remote_directory/'request.json';ack=a.remote_directory/'ack.json'
    read="import pathlib,json,sys; p=pathlib.Path(sys.argv[1]); ack=p.with_name('ack.json'); print(json.dumps(dict(request=json.loads(p.read_text()),ack=json.loads(ack.read_text()) if ack.exists() else None)) if p.exists() else '')"
    deadline=time.monotonic()+a.timeout
    while time.monotonic()<deadline:
        try:
            output=subprocess.check_output(ssh+[shlex.join([a.remote_python,'-c',read,str(request)])],text=True,timeout=60)
        except (subprocess.CalledProcessError,subprocess.TimeoutExpired) as error:
            print('Audit polling will retry: '+str(error),flush=True);time.sleep(a.poll_seconds);continue
        if not output.strip():time.sleep(a.poll_seconds);continue
        message=json.loads(output);data=message['request']
        if (message.get('ack') or {}).get('nonce')==data['nonce']:
            time.sleep(a.poll_seconds);continue
        actual=fingerprint(a.source.resolve(),a.calibration.resolve());expected=data['live_source_hashes']
        changed=sorted(k for k in actual.keys()|expected.keys() if actual.get(k)!=expected.get(k))
        result=dict(nonce=data['nonce'],approved=not changed,source_identity=identity(actual),changed=changed,
                    authoritative_source=str(a.source.resolve()),checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        write="import pathlib,sys; p=pathlib.Path(sys.argv[1]); t=p.with_suffix('.tmp'); t.write_text(sys.stdin.read()); t.replace(p)"
        try:
            subprocess.run(ssh+[shlex.join([a.remote_python,'-c',write,str(ack)])],input=json.dumps(result,indent=2)+'\n',text=True,check=True,timeout=60)
        except (subprocess.CalledProcessError,subprocess.TimeoutExpired) as error:
            print('Audit acknowledgment will retry: '+str(error),flush=True);time.sleep(a.poll_seconds);continue
        print(json.dumps(result,indent=2),flush=True)
        if changed:raise RuntimeError('Live source was refitted; refresh and recollect before training')
        return result
    raise TimeoutError('Remote trainer did not request a live-source audit')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--host',required=True);p.add_argument('--ssh-config',type=Path,default=Path.home() / '.ssh/config')
    p.add_argument('--source',type=Path,required=True);p.add_argument('--calibration',type=Path,required=True)
    p.add_argument('--poll-seconds',type=float,default=30);p.add_argument('--remote-directory',type=Path,required=True);p.add_argument('--remote-python',default='/tmp/juggling-newton-venv/bin/python');p.add_argument('--timeout',type=float,default=172800)
    audit(p.parse_args())
