"""Export format regression using synthetic weights, never a deployable policy."""
import sys,tempfile,subprocess,json
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from export_policy import export
from training_data import output_baseline

def check(eigen):
    rng=np.random.default_rng(19);x=rng.normal(size=40).astype('float32')
    arrays=dict(input_mean=rng.normal(size=40).astype('float32'),input_scale=rng.uniform(.5,2,40).astype('float32'),output_mean=rng.normal(size=84).astype('float32'),output_scale=rng.uniform(.5,2,84).astype('float32'))
    sizes=[40,256,256,256,84]
    arrays['output_baseline']=output_baseline([0.,.03,.062,.097])
    expected=(x-arrays['input_mean'])/arrays['input_scale']
    for i,(ni,no) in enumerate(zip(sizes[:-1],sizes[1:])):
        arrays[f'weight_{i}']=rng.normal(0,.05,(ni,no)).astype('float32')
        arrays[f'bias_{i}']=rng.normal(0,.05,no).astype('float32')
        expected=expected@arrays[f'weight_{i}']+arrays[f'bias_{i}']
        if i<3:expected=np.tanh(expected)
    expected=expected*arrays['output_scale']+arrays['output_mean']+x@arrays['output_baseline']
    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp);np.savez(root/'synthetic.npz',**arrays);export(root/'synthetic.npz',root)
        (root/'check.cpp').write_text('#include "ampc_policy.hpp"\n#include <iostream>\n#include <iomanip>\nint main(){Eigen::Matrix<float,40,1> x; for(int i=0;i<40;++i)std::cin>>x[i];auto y=juggling_ampc::predict(x);std::cout<<std::setprecision(10);for(int i=0;i<84;++i)std::cout<<y[i]<<" ";}\n')
        subprocess.run(['c++','-O3','-std=c++17','-I'+str(eigen),str(root/'check.cpp'),'-o',str(root/'check')],check=True)
        out=subprocess.check_output([str(root/'check')],input=' '.join(map(str,x)),text=True);actual=np.fromstring(out,sep=' ')
    error=float(np.max(abs(actual-expected)));report=dict(synthetic_weights_only=True,layer_sizes=sizes,max_abs_error=error,passed=error<2e-6)
    print(json.dumps(report,indent=2));assert report['passed']
if __name__=='__main__':check(Path(sys.argv[1]))
