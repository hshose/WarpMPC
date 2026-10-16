"""Offline check of the actual compiled C++ controller input/output contract."""
import bootstrap
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from cpp.python_controller import NativeController
from check_policy_export import inference_error


def check(checkpoint,profile,samples=256):
    weights=np.load(checkpoint);digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    manifest=json.loads((profile/'policy/policy_manifest.json').read_text())
    if manifest['checkpoint_sha256']!=digest:raise ValueError('Compiled profile belongs to another checkpoint')
    if manifest.get('hidden_layers')!=[256,256,256]:raise ValueError('Expected the requested three 256-neuron hidden layers')
    arrays=json.loads((profile/'factors/metadata.json').read_text())['arrays']
    rng=np.random.default_rng(994)
    inputs=(weights['input_mean']+rng.uniform(-2,2,(samples,40))*weights['input_scale']).astype(np.float32)
    # Exercise changing references while respecting the native reference setter.
    inputs[:,30:32]/=np.maximum(np.linalg.norm(inputs[:,30:32],axis=1,keepdims=True)/.30,1)
    inputs[:,32]=np.clip(inputs[:,32],-.049,.049)
    home=np.asarray(arrays['home']);lower=np.maximum(arrays['joint_lower'],home-.99)+1e-4
    upper=np.minimum(arrays['joint_upper'],home+.99)-1e-4
    inputs[:,33:]=np.clip(inputs[:,33:],lower,upper)
    expected=(inputs-weights['input_mean'])/weights['input_scale']
    for i in range(4):
        expected=expected@weights[f'weight_{i}']+weights[f'bias_{i}']
        if i<3:expected=np.tanh(expected)
    expected=expected*weights['output_scale']+weights['output_mean']+inputs@weights['output_baseline']
    controller=NativeController(profile/'build/controller/libjuggling_controller.so',profile/'factors',compiled=True)
    if len(weights['output_mean'])==7:
        import casadi as ca
        mass=ca.Function.load(str(profile/'factors/panda_mass.casadi'))
        bias=ca.Function.load(str(profile/'factors/panda_bias.casadi'))
    actual=[];elapsed=[];iterations=[]
    try:
        for row in inputs:
            controller.set_reference(row[30:33],row[33:])
            plan,_,stats=controller.step(row[:23],row[23:30],.02)
            if len(weights['output_mean'])==7:
                q,dq,tau=plan[1,6:13],plan[1,13:20],plan[1,23:30]
                actual.append(np.linalg.solve(np.asarray(mass(q)),tau-np.asarray(bias(q,dq)).ravel()))
            else:actual.append(np.concatenate([plan[:4,6:13],plan[:4,13:20],plan[:4,23:30]],axis=1).reshape(84))
            elapsed.append(float(stats[0])*1000);iterations.append([int(stats[12]),int(stats[17])])
    finally:controller.close()
    report=dict(checkpoint_sha256=digest,**inference_error(np.asarray(actual),expected,inputs,weights),
        samples=samples,solver_iterations=sorted(set(tuple(v) for v in iterations)),
        controller_step_median_us=float(np.median(elapsed)),controller_step_p99_us=float(np.quantile(elapsed,.99)),
        hardware_executed=False,inputs='normalization-range probes with admissible moving references')
    report['passed']=bool(report['passed'] and report['solver_iterations']==[(0,0)])
    (profile/'controller_policy_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    if not report['passed']:raise RuntimeError('Compiled controller differs from trained policy')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('checkpoint',type=Path);p.add_argument('profile',type=Path);p.add_argument('--samples',type=int,default=256);a=p.parse_args()
    if a.samples<1:raise ValueError('Positive sample count required')
    check(a.checkpoint,a.profile,a.samples)
