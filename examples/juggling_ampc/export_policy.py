"""Export trained normalization and weights to allocation-free Eigen inference."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np


def export(checkpoint,output):
    d=np.load(checkpoint);layers=len([k for k in d.files if k.startswith('weight_')]);nx=len(d['input_mean']);ny=len(d['output_mean'])
    if nx!=40 or ny not in (7,84):raise ValueError('Hardware contract requires 40 inputs and seven accelerations (or legacy 84-node outputs)')
    sizes=[40,256,256,256,ny]
    if layers!=4 or any(d[f'weight_{i}'].shape!=(ni,no) or d[f'bias_{i}'].shape!=(no,) for i,(ni,no) in enumerate(zip(sizes[:-1],sizes[1:]))):
        raise ValueError('Export requires exactly three hidden layers of 256 neurons')
    for key,n in [('input_mean',nx),('input_scale',nx),('output_mean',ny),('output_scale',ny)]:
        if d[key].shape!=(n,):raise ValueError('Invalid normalization shape: '+key)
        if key.endswith('scale') and np.any(d[key]<=0):raise ValueError('Normalization scales must be positive')
    lines=['#pragma once','#include <Eigen/Dense>','#include <cmath>','#include <stdexcept>','namespace juggling_ampc {',f'inline constexpr int input_size={nx}, output_size={ny};']
    def array(name,value):
        value=np.asarray(value,dtype=np.float32)
        if not np.isfinite(value).all():raise ValueError('Nonfinite weights')
        literal=[]
        for v in value.ravel():
            s=format(float(v),'.9g');s=s if '.' in s or 'e' in s else s+'.0';literal.append(s+'f')
        lines.append(f'inline constexpr float {name}[]={{'+','.join(literal)+'};')
    for key in ['input_mean','input_scale','output_mean','output_scale']:array(key,d[key])
    baseline=d['output_baseline'] if 'output_baseline' in d else np.zeros((nx,ny),np.float32)
    if baseline.shape!=(nx,ny):raise ValueError('Invalid affine output baseline')
    array('output_baseline',baseline)
    for i in range(layers):array(f'w{i}',d[f'weight_{i}']);array(f'b{i}',d[f'bias_{i}'])
    lines+=['inline Eigen::Matrix<float,output_size,1> predict(const Eigen::Matrix<float,input_size,1>& input) {',
        'if(!input.allFinite()) throw std::runtime_error("Nonfinite AMPC input");',
        'Eigen::Matrix<float,input_size,1> x0;',
        'for(int i=0;i<input_size;++i)x0[i]=(input[i]-input_mean[i])/input_scale[i];']
    for i in range(layers):
        ni,no=d[f'weight_{i}'].shape
        lines += [f'Eigen::Matrix<float,{no},1> x{i+1};',f'x{i+1}.noalias()=Eigen::Map<const Eigen::Matrix<float,{ni},{no},Eigen::RowMajor>>(w{i}).transpose()*x{i};',f'x{i+1}+=Eigen::Map<const Eigen::Matrix<float,{no},1>>(b{i});']
        if i<layers-1:lines +=[f'x{i+1}=x{i+1}.array().tanh().matrix();']
    lines += [f'for(int i=0;i<output_size;++i)x{layers}[i]=x{layers}[i]*output_scale[i]+output_mean[i];',f'x{layers}.noalias()+=Eigen::Map<const Eigen::Matrix<float,{nx},{ny},Eigen::RowMajor>>(output_baseline).transpose()*input;',f'if(!x{layers}.allFinite())throw std::runtime_error("Nonfinite AMPC output");',f'return x{layers};','}','}']
    output.mkdir(parents=True,exist_ok=True);(output/'ampc_policy.hpp').write_text('\n'.join(lines)+'\n')
    metadata=dict(checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),inputs=40,outputs=ny,hidden_layers=[256,256,256],activation='tanh',precision='float32')
    metadata['output_representation']='first future knot projected joint acceleration' if ny==7 else 'normalized residual plus fixed affine baseline'
    training=checkpoint.parent/'training.json'
    if training.exists():metadata['mpc_spec']=json.loads(training.read_text())['mpc_spec']
    (output/'policy_manifest.json').write_text(json.dumps(metadata,indent=2)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('checkpoint',type=Path);p.add_argument('output',type=Path);a=p.parse_args();export(a.checkpoint,a.output)
