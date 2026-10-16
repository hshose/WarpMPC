"""Hash the live simulator/calibration again before collecting and before training."""
import argparse,hashlib,json
from pathlib import Path


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def fingerprint(source,calibration):
    paths=[]
    for directory in ('newton_sim','identification','estimation','tracking'):
        paths.extend(p for p in (source/directory).rglob('*.py') if '__pycache__' not in p.parts)
    paths.extend([calibration,source/'config/cpp_ball_spec.json'])
    # Both user-provided artifacts and fit provenance must remain current.
    provenance=Path(__file__).parent/'calibration/provenance.json'
    if provenance.exists():
        paths.extend(source/p for p in json.loads(provenance.read_text())['files'] if (source/p).exists())
    return {str(p.relative_to(source)):digest(p) for p in sorted(set(paths))}

def assert_current(source,calibration,manifest):
    actual=fingerprint(source,calibration)
    if actual != manifest:
        changed=sorted(k for k in actual.keys()|manifest.keys() if actual.get(k)!=manifest.get(k))
        raise RuntimeError('Live calibration/simulator changed; refresh snapshot and recollect before training: '+', '.join(changed))
    return actual

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--calibration',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--check',action='store_true');a=p.parse_args()
    current=fingerprint(a.source.resolve(),a.calibration.resolve())
    if a.check:assert_current(a.source.resolve(),a.calibration.resolve(),json.loads(a.manifest.read_text()))
    else:a.manifest.write_text(json.dumps(current,indent=2)+'\n')
