"""Compile each distinct generated C function once, then link a variant runtime."""
import argparse,concurrent.futures,hashlib,json,os,shutil,subprocess
from pathlib import Path


def build(directory,output,jobs=32):
    root=Path.cwd();directory=Path(directory).resolve();output=Path(output).resolve()
    cc=shutil.which(os.environ.get('CC','clang'));cxx=shutil.which(os.environ.get('CXX','clang++'))
    if not cc or not cxx:raise RuntimeError('Clang compilers required')
    version=subprocess.check_output([cc,'--version'],text=True)
    # CPU identity is included because object code uses -march=native.
    machine='\n'.join(line for line in Path('/proc/cpuinfo').read_text().split('\n\n')[0].splitlines()
                      if line.split(':')[0].strip() in ['vendor_id','model name','cpu family','model','stepping','flags'])
    cache=root/'build/variant_objects';cache.mkdir(parents=True,exist_ok=True)
    functions=json.loads((directory/'metadata.json').read_text())['functions']
    baseline=root/'build/cpp_factors'
    shutil.copyfile(baseline/'libjuggling_factors.so',directory/'libjuggling_factors.so')
    def compile_one(name):
        source=directory/(name+'.c');key=hashlib.sha256(source.read_bytes()+version.encode()+machine.encode()+b'O3,NDEBUG,native').hexdigest()
        old=baseline/source.name
        if old.exists() and old.read_bytes()==source.read_bytes():
            # A previously modified function must not shadow the baseline symbol.
            (directory/(name+'.so')).unlink(missing_ok=True)
            return None
        target=cache/(key+'.o')
        if not target.exists():
            temporary=cache/(key+f'.{os.getpid()}.o')
            subprocess.run([cc,'-O3','-DNDEBUG','-march=native','-fPIC','-c',str(source),'-o',str(temporary)],check=True)
            temporary.replace(target)
        subprocess.run([cc,'-shared',str(target),'-o',str(directory/(name+'.so')),'-lm'],check=True)
        return target
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as executor:objects=list(executor.map(compile_one,functions))
    import casadi
    cmake=root/'.venv/bin/cmake';casadi_dir=Path(casadi.__file__).parent/'cmake'
    subprocess.run([str(cmake),'-S','optimization/native','-B',str(output),'-DCMAKE_C_COMPILER='+cc,'-DCMAKE_CXX_COMPILER='+cxx,
        '-Dcasadi_DIR='+str(casadi_dir),'-Dosqp_DIR='+str(root/'build/osqp_install/lib/cmake/osqp'),'-DFACTOR_DIR='+str(directory),'-DCMAKE_EXPORT_COMPILE_COMMANDS=ON'],check=True)
    subprocess.run([str(cmake),'--build',str(output),'-j',str(jobs)],check=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('factors');p.add_argument('output');p.add_argument('--jobs',type=int,default=32);a=p.parse_args();build(a.factors,a.output,a.jobs)
