"""Compile the independently exported native C factors for baseline rollouts."""
import argparse, concurrent.futures, hashlib, json, os, shutil, subprocess
from pathlib import Path


def build(a):
    compiler = shutil.which(a.cc)
    if not compiler:
        raise RuntimeError(f'C compiler unavailable: {a.cc}')
    root = a.directory.resolve()
    names = json.loads((root / 'metadata.json').read_text())['functions']
    version = subprocess.check_output([compiler, '--version'], text=True)
    flags = ['-O3', '-DNDEBUG', '-march=native', '-fPIC', '-shared']
    def compile_one(name):
        source = root / (name + '.c')
        output = root / (name + '.so')
        stamp = root / (name + '.build.json')
        record = dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), compiler=version, flags=flags)
        if output.exists() and stamp.exists() and json.loads(stamp.read_text()) == record:
            return
        temporary = root / (name + f'.{os.getpid()}.so')
        subprocess.run([compiler, *flags, str(source), '-o', str(temporary), '-lm'], check=True)
        temporary.replace(output)
        stamp.write_text(json.dumps(record, indent=2) + '\n')
        print('Compiled ' + name, flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.jobs) as pool:
        list(pool.map(compile_one, names))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('directory', type=Path)
    p.add_argument('--cc', default=os.environ.get('CC', 'cc'))
    p.add_argument('--jobs', type=int, default=8)
    build(p.parse_args())
