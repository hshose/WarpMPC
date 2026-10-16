"""Stage six private H100 cases; submit only from an already uploaded snapshot.

Run with the CLAIX system Python (python3 -S). Dependencies must have already
been copied under the sweep root. This script performs no GPU computation.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--archive',type=Path,required=True)
    p.add_argument('--factors',type=Path,required=True)
    p.add_argument('--source-revision',required=True)
    p.add_argument('--reference-revision',required=True)
    p.add_argument('--seconds',type=int,nargs='+',choices=(5,10,20),default=[5])
    p.add_argument('--job-prefix',default='juggle')
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--cache-root',type=Path,help='Copy each prior case cache into its new private directory')
    a=p.parse_args();a.root.mkdir(parents=True,exist_ok=True)
    dependency=a.root/'dependencies'
    assert (dependency/'venv/bin/python').exists()
    assert (dependency/'external').is_dir()
    archive_hash=hashlib.sha256(a.archive.read_bytes()).hexdigest()
    submission={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'reference_commit':a.reference_revision,
                'source_revision':a.source_revision,'source_archive_sha256':archive_hash,
                'factor_files_sha256':{str(f.relative_to(a.factors)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(a.factors.rglob('*')) if f.is_file()},
                'jobs':[]}
    for protocol in ('arm15','calibrated'):
        for seconds in a.seconds:
            case=a.root/f'{protocol}_{seconds}s'
            source=case/'source'
            source.mkdir(parents=True,exist_ok=False)
            with tarfile.open(a.archive) as archive:archive.extractall(source)
            shutil.copytree(dependency/'external',source/'external',symlinks=False)
            shutil.copytree(a.factors,case/'gpu_factors')
            (case/'results').mkdir()
            if a.cache_root:
                for kind in ('jax','warp','cuda'):
                    previous=a.cache_root/case.name/'cache'/kind
                    if previous.is_dir():shutil.copytree(previous,case/'cache'/kind)

            expected=(source/'examples/juggling_ampc/reference/config/cpp_ball_spec.json').read_bytes()
            actual=json.loads((case/'gpu_factors/metadata.json').read_text())['spec']
            assert json.loads(expected)==actual,'Stale generated formulation'
            manifest=json.loads((source/'examples/juggling_ampc/source_manifest.json').read_text())
            assert manifest['formulation_refresh']['revision']==a.reference_revision,'Wrong pinned reference revision'
            record={'protocol':protocol,'seconds':seconds,'case':str(case),
                    'reference_commit':a.reference_revision,
                    'source_revision':a.source_revision,'source_archive_sha256':archive_hash,
                    'spec_sha256':hashlib.sha256(expected).hexdigest(),
                    'collection_multipliers':{'ball_flight':1.,'ball_contact':1.,'sensing':1.,'plan_delivery':1.,'arm_following':1.5 if protocol=='arm15' else 1.},
                    'validation_multipliers':'all calibrated (1.0)',
                    'jax_preallocation':False,'jax_allocator':'cuda_async',
                    'cache_copied_from':str(a.cache_root/case.name/'cache') if a.cache_root else None,
                    'batch':60000,'epochs':100}
            command=['sbatch','--parsable','--job-name='+f'{a.job_prefix}_{protocol}_{seconds}s',
                     '--output='+str(case/'results/slurm-%j.log'),
                     str(source/'examples/juggling_ampc/run_claix23_instant.sh'),
                     str(case),str(seconds),protocol,str(dependency)]
            record['command']=command
            if not a.prepare_only:
                result=subprocess.run(command,check=True,capture_output=True,text=True)
                record.update(job_id=result.stdout.strip().split(';')[0],stdout=result.stdout,stderr=result.stderr)
            (case/'experiment.json').write_text(json.dumps(record,indent=2)+'\n')
            submission['jobs'].append(record)
            (a.root/'submissions.json').write_text(json.dumps(submission,indent=2)+'\n')
            print(json.dumps(record),flush=True)


if __name__=='__main__':main()
