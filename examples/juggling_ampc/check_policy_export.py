"""Check actual Eigen inference against checkpoint math and measure CPU latency."""
import argparse,hashlib,json,subprocess,tempfile
from pathlib import Path
import numpy as np


def inference_error(actual,expected,inputs,weights):
    difference=abs(np.asarray(actual,dtype=float)-np.asarray(expected,dtype=float))
    # The affine baseline and final addition round in physical units. Dividing
    # their last-bit error by a tiny residual scale can falsely reject an exact
    # export; keep a separate bounded float32 rounding allowance for those ops.
    magnitude=abs(expected)+abs(weights['output_mean'])+abs(inputs).astype(float)@abs(weights['output_baseline']).astype(float)
    tolerance=1e-4*weights['output_scale']+4*np.finfo(np.float32).eps*magnitude
    return dict(passed=bool(np.isfinite(actual).all() and np.all(difference<=tolerance)),
        max_abs_error=float(difference.max()),
        max_normalized_error=float((difference/weights['output_scale']).max()),
        max_tolerance_ratio=float((difference/tolerance).max()),
        normalized_tolerance=1e-4,physical_roundoff_epsilon_multiplier=4)


def check(a):
    weights=np.load(a.checkpoint)
    checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()
    if checkpoint_sha256!=json.loads((a.export/'policy_manifest.json').read_text())['checkpoint_sha256']:raise ValueError('Export belongs to a different checkpoint')
    # Use physical held-out states when supplied; otherwise exercise the learned
    # normalization range. Neither path constitutes a closed-loop qualification.
    if a.inputs:
        data=np.load(a.inputs,mmap_mode='r');inputs=np.asarray(data['inputs'][:a.samples],np.float32)
    else:
        rng=np.random.default_rng(611)
        inputs=(rng.uniform(-2,2,(a.samples,40))*weights['input_scale']+weights['input_mean']).astype(np.float32)
    expected=(inputs-weights['input_mean'])/weights['input_scale']
    for i in range(4):
        expected=expected@weights[f'weight_{i}']+weights[f'bias_{i}']
        if i<3:expected=np.tanh(expected)
    expected=expected*weights['output_scale']+weights['output_mean']+inputs@weights['output_baseline']
    source=r'''
#define EIGEN_RUNTIME_NO_MALLOC
#include "ampc_policy.hpp"
#include <chrono>
#include <iostream>
#include <vector>
#include <algorithm>
int main(int argc,char** argv){
 const int n=std::stoi(argv[1]);
 std::vector<float> inputs(n*40),outputs(n*juggling_ampc::output_size);
 std::vector<double> times(n);
 std::cin.read(reinterpret_cast<char*>(inputs.data()),inputs.size()*sizeof(float));
 if(!std::cin)return 2;
 Eigen::internal::set_is_malloc_allowed(false);
 for(int i=0;i<n;++i){
  Eigen::Matrix<float,40,1> input=Eigen::Map<const Eigen::Matrix<float,40,1>>(inputs.data()+40*i);
  auto start=std::chrono::steady_clock::now();
  auto result=juggling_ampc::predict(input);
  times[i]=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count();
  Eigen::Map<Eigen::Matrix<float,juggling_ampc::output_size,1>>(outputs.data()+juggling_ampc::output_size*i)=result;
 }
 Eigen::internal::set_is_malloc_allowed(true);
 std::cout.write(reinterpret_cast<char*>(outputs.data()),outputs.size()*sizeof(float));
 std::sort(times.begin(),times.end());
 std::cerr<<"{\"median_us\":"<<times[n/2]<<",\"p99_us\":"<<times[std::min(n-1,int(.99*n))]<<"}";
}
'''
    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp);(root/'check.cpp').write_text(source)
        subprocess.run(['c++','-O3','-march=native','-std=c++17','-I'+str(a.eigen.resolve()),'-I'+str(a.export.resolve()),str(root/'check.cpp'),'-o',str(root/'check')],check=True)
        output=subprocess.run([str(root/'check'),str(len(inputs))],input=inputs.tobytes(),stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True)
    actual=np.frombuffer(output.stdout,np.float32).reshape(expected.shape)
    report=dict(checkpoint_sha256=checkpoint_sha256,**inference_error(actual,expected,inputs,weights),samples=len(inputs),eigen_heap_allocations_allowed=False,latency=json.loads(output.stderr),inputs='supplied physical states' if a.inputs else 'synthetic normalization-range probes')
    (a.export/'eigen_validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
    if not report['passed']:raise RuntimeError('Eigen export differs from trained checkpoint')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('checkpoint',type=Path);p.add_argument('export',type=Path);p.add_argument('--eigen',type=Path,required=True);p.add_argument('--inputs',type=Path);p.add_argument('--samples',type=int,default=1024);a=p.parse_args()
    if a.samples<1:raise ValueError('Positive sample count required')
    check(a)
