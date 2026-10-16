"""Stage private CLAIX source snapshots and submit H100 training (5 s by default)."""
import argparse
import datetime
import os
import json
from pathlib import Path
import shutil
import subprocess
import tarfile


def main():
    repo = Path(__file__).resolve().parents[2]
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dependencies', type=Path, required=True,
                   help='Installed dependency directory containing venv/, external/, eigen/')
    p.add_argument('--root', type=Path,
                   default=repo / 'results/juggling_ampc' / stamp)
    p.add_argument('--archive', type=Path, help='Source tar.gz; defaults to git archive HEAD')
    p.add_argument('--factors', type=Path, help='Matching generated GPU factors; otherwise generated in job')
    p.add_argument('--seconds', type=int, nargs='+', choices=(5, 10, 20), default=[5])
    p.add_argument('--protocol', choices=('arm15', 'calibrated'), default='arm15')
    p.add_argument('--account', default=os.environ.get('SBATCH_ACCOUNT'))
    p.add_argument('--partition', default=os.environ.get('SBATCH_PARTITION'))
    p.add_argument('--prepare-only', action='store_true')
    a = p.parse_args()
    a.root = a.root.resolve()
    a.dependencies = a.dependencies.resolve()
    if not (a.dependencies / 'venv/bin/python').exists():
        p.error('--dependencies must contain venv/bin/python')
    a.root.mkdir(parents=True, exist_ok=True)
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    archive = a.archive.resolve() if a.archive else a.root / 'source.tar.gz'
    if not a.archive:
        subprocess.run(['git', 'archive', '--format=tar.gz', '-o', str(archive), 'HEAD'],
                       cwd=repo, check=True)
    records = []
    for seconds in a.seconds:
        case = a.root / f'{a.protocol}_{seconds}s'
        source = case / 'source'
        source.mkdir(parents=True, exist_ok=False)
        with tarfile.open(archive) as bundle:
            bundle.extractall(source, filter='data')
        # Every job has private editable sources; installed packages are shared.
        shutil.copytree(a.dependencies / 'external', source / 'external', dirs_exist_ok=True)
        if a.factors:
            shutil.copytree(a.factors, case / 'gpu_factors')
            spec = json.loads((source / 'examples/juggling_ampc/reference/config/cpp_ball_spec.json').read_text())
            metadata = json.loads((case / 'gpu_factors/metadata.json').read_text())
            if metadata['spec'] != spec:
                raise ValueError('Provided GPU factors do not match the submitted formulation')
        results = case / 'results'
        results.mkdir()
        command = ['sbatch', '--parsable',
                   '--job-name=' + f'juggling_{a.protocol}_{seconds}s',
                   '--output=' + str(results / 'slurm-%j.log'),
                   str(source / 'examples/juggling_ampc/run_h100_ampc.sh'),
                   str(case), str(seconds), a.protocol, str(a.dependencies)]
        if a.account:
            command.insert(2, '--account=' + a.account)
        if a.partition:
            command.insert(2, '--partition=' + a.partition)
        record = dict(seconds=seconds, protocol=a.protocol, source_revision=revision,
                      source_archive=str(archive), results=str(results), command=command)
        if not a.prepare_only:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            record['job_id'] = result.stdout.strip().split(';')[0]
        records.append(record)
        (a.root / 'submissions.json').write_text(json.dumps(records, indent=2) + '\n')
        print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
