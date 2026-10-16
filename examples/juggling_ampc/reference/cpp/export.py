"""Export the unchanged scaled factors; deployment/evaluation needs no Python."""
import argparse,json,os,tempfile
from dataclasses import asdict
from pathlib import Path
import casadi as ca
import numpy as np
from arm.model import Panda,HOME,TORQUE_LIMIT,VELOCITY_LIMIT,ACCEL_LIMIT,TORQUE_RATE_LIMIT,JERK_LIMIT
from arm.spin_formulation import Config,create_factors,bounds,NX,NZ,NP,SCALE,OFFSET


def export(directory):
    directory=Path(directory).resolve();directory.mkdir(parents=True,exist_ok=True)
    cfg=Config(actuator_margin=.98);robot=Panda()
    first,stage,last,diagnostic,lo,hi,groups=create_factors(cfg,robot)
    p=ca.MX.sym('p',NP);functions=[];counts={}
    for name,f,nvar in [('first',first,NZ),('stage',stage,2*NZ),('last',last,NZ)]:
        z=ca.MX.sym('z',nvar)
        if name=='first':r=ca.MX.zeros(0,1);c=f(z,p)
        elif name=='stage':r,c=f(z[:NZ],z[NZ:],p)
        else:r,c=f(z,p)
        r=np.sqrt(cfg.objective_scale)*r
        jr=ca.jacobian(r,z);jc=ca.jacobian(c,z)
        functions += [ca.Function(name+'_data',[z,p],[r,jr,c,jc]),ca.Function(name+'_value',[z,p],[r,c])]
        counts[name]=dict(variables=nvar,residuals=r.numel(),constraints=c.numel(),jr_nnz=jr.nnz(),jc_nnz=jc.nnz())
    functions += [robot.mass,robot.bias,robot.rnea,robot.fk,robot.contact_fk]
    cwd=Path.cwd()
    # Preserve timestamps of unchanged generated files, avoiding an expensive
    # recompile when only the runtime/harness is rebuilt.
    with tempfile.TemporaryDirectory(dir=directory) as temporary:
        try:
            os.chdir(temporary)
            for f in functions:
                f.save(f.name()+'.casadi')
                cg=ca.CodeGenerator(f.name()+'.c',{'with_header':True});cg.add(f);cg.generate()
        finally:os.chdir(cwd)
        for source in Path(temporary).iterdir():
            target=directory/source.name;data=source.read_bytes()
            if not target.exists() or target.read_bytes()!=data:target.write_bytes(data)
    lb,ub=bounds(cfg,robot)
    arrays=dict(scale=SCALE,offset=OFFSET,lower=lb,upper=ub,stage_lower=lo,stage_upper=hi,
                home=HOME,origin=robot.origin,torque_limit=TORQUE_LIMIT,velocity_limit=VELOCITY_LIMIT,
                acceleration_limit=ACCEL_LIMIT,torque_rate_limit=TORQUE_RATE_LIMIT,jerk_limit=JERK_LIMIT,
                joint_lower=robot.q_min,joint_upper=robot.q_max)
    lines=['#pragma once','#include <array>','#include <cmath>','namespace juggling::generated {',
           f'inline constexpr int NX={NX}, NZ={NZ}, NP={NP}, N={cfg.horizon};']
    for name,value in asdict(cfg).items():
        if isinstance(value,(int,float)) and not isinstance(value,bool):lines.append(f'inline constexpr double {name}={value!r};')
    for name,values in arrays.items():
        def number(v):return 'INFINITY' if np.isposinf(v) else '-INFINITY' if np.isneginf(v) else repr(float(v))
        lines.append(f'inline const std::array<double,{len(values)}> {name}={{{", ".join(map(number,values))}}};')
    lines += [f'inline constexpr int contact_begin={groups["contact"][0]}, contact_end={groups["contact"][1]};',
              f'inline constexpr int NF={counts["first"]["constraints"]}, NC={len(lo)}, NL=1;',
              f'inline constexpr const char * mass_name="{robot.mass.name()}", *bias_name="{robot.bias.name()}", *rnea_name="{robot.rnea.name()}", *fk_name="{robot.fk.name()}", *contact_fk_name="{robot.contact_fk.name()}";', '}']
    (directory/'model_config.hpp').write_text('\n'.join(lines)+'\n')
    metadata=dict(casadi_version=ca.__version__,config=asdict(cfg),counts=counts,groups=groups,
                  functions=[f.name() for f in functions],arrays={n:v.tolist() for n,v in arrays.items()})
    (directory/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(dict(directory=str(directory),counts=counts,functions=metadata['functions']),indent=2))


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default='build/cpp_factors');args=p.parse_args();export(args.output)


if __name__=='__main__':main()
