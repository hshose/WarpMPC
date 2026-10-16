"""Reuse a complete parametric build after checking the ABI and source identity."""
import argparse,hashlib,json,os,shutil,subprocess,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def source_hash():
    paths=set()
    for directory in ['cpp','optimization/native']:
        for p in (ROOT/directory).rglob('*'):
            if p.suffix in ('.cpp','.hpp','.h','.inc') or p.name=='CMakeLists.txt':paths.add(p)
    h=hashlib.sha256()
    for p in sorted(paths):h.update(str(p.relative_to(ROOT)).encode());h.update(p.read_bytes())
    return h.hexdigest()
def stamp(profile):
    (profile/'runtime_source.sha256').write_text(source_hash()+'\n')
def manifest(profile):
    files=[]
    for directory in ['factors','build']:
        for base,dirs,names in os.walk(profile/directory,followlinks=True):
            for name in names:
                p=Path(base)/name
                if p.suffix=='.so' or name in ['model_config.hpp','metadata.json','runtime_values.txt','juggling_ball_cpp','juggling_franka_session','juggling_vicon']:files.append(p)
    files += [profile/n for n in ['spec.json','juggling_home.json','runtime_source.sha256','source_snapshot.tar.gz'] if (profile/n).exists()]
    (profile/'manifest.sha256').write_text(''.join(digest(p)+'  '+str(p.relative_to(profile))+'\n' for p in sorted(files)))
def reuse(profile,source):
    start=time.monotonic();old=json.loads((source/'factors/metadata.json').read_text());new=json.loads((profile/'factors/metadata.json').read_text())
    if not new.get('runtime_schema') or new['runtime_schema']!=old.get('runtime_schema'):return False
    if not (source/'runtime_source.sha256').exists() or (source/'runtime_source.sha256').read_text().strip()!=source_hash():return False
    # Never reuse a function whose graph changed, even if dimensions still match.
    for name in new['functions']:
        if digest(profile/'factors'/(name+'.c'))!=digest(source/'factors'/(name+'.c')):return False
    subprocess.run(['sha256sum','--quiet','-c','manifest.sha256'],cwd=source,check=True)
    if (profile/'build').exists():raise ValueError('New profile already has a build directory')
    (profile/'build').symlink_to((source/'build').resolve(),target_is_directory=True)
    for p in list((profile/'factors').iterdir()):
        old=source/'factors'/p.name
        if p.suffix in ('.c','.h','.casadi') and old.exists() and digest(p)==digest(old):
            p.unlink();p.symlink_to(old.resolve())
    if (source/'source_snapshot.tar.gz').exists():
        (profile/'source_snapshot.tar.gz').symlink_to((source/'source_snapshot.tar.gz').resolve())
    for p in (source/'factors').glob('*.so'):(profile/'factors'/p.name).symlink_to(p.resolve())
    # Binary/header hashes describe the immutable generic program, while the
    # separately hashed runtime_values and spec describe this numerical model.
    shutil.copyfile(source/'factors/model_config.hpp',profile/'factors/model_config.hpp')
    env=dict(os.environ,JUGGLING_RUNTIME_VALUES=str(profile/'factors/runtime_values.txt'),OPENBLAS_NUM_THREADS='1',JUGGLING_EVAL_THREADS='4')
    checks=[('runtime_description.json',['build/runtime/juggling_ball_cpp','--describe']),
            ('native_description.json',['build/session/juggling_franka_session','--describe']),
            ('session_checks.json',['build/controller/juggling_session_check']),
            ('runtime_checks.json',['build/runtime/juggling_ball_cpp','--self-check','--factors',str(profile/'factors')])]
    if (profile/'juggling_home.json').exists():checks[-1][1].extend(['--home-config',str(profile/'juggling_home.json')])
    for name,args in checks:
        with (profile/name).open('w') as f:subprocess.run([str(profile/args[0]),*args[1:]],env=env,stdout=f,check=True)
    shutil.copyfile(source/'estimator_checks.json',profile/'estimator_checks.json')
    stamp(profile);manifest(profile)
    result=dict(reused_binaries=True,source_profile=str(source),preparation_s=time.monotonic()-start,
      runtime_values_sha256=digest(profile/'factors/runtime_values.txt'),physical_robot_used=False)
    (profile/'runtime_reuse.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
    return True

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('profile',type=Path);p.add_argument('--source',type=Path);p.add_argument('--stamp',action='store_true');a=p.parse_args()
    if a.stamp:stamp(a.profile)
    elif not reuse(a.profile.resolve(),a.source.resolve()):raise SystemExit(20)
