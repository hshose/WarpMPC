"""Native factors for independent pruning/mesh experiments; baseline stays intact."""
import argparse,json,os,tempfile
from functools import partial
from dataclasses import asdict, replace
from pathlib import Path
import casadi as ca
import numpy as np
from arm.model import Panda,HOME,TORQUE_LIMIT,VELOCITY_LIMIT,ACCEL_LIMIT,TORQUE_RATE_LIMIT,JERK_LIMIT
from arm.spin_formulation import Config,NX,NZ,NP,SCALE,OFFSET,bounds
from optimization.pruning.formulation import create_factors
from optimization.mesh.formulation import Config as MeshConfig,grid
from optimization import runtime_parameters as rt


def export(spec,directory):
    directory=Path(directory).resolve();directory.mkdir(parents=True,exist_ok=True)
    cfg=Config(**dict({'actuator_margin':.98},**spec.get('config',{})));robot=Panda(assembly=spec.get('assembly'),**spec.get('geometry',{}))
    if cfg.horizon<3 or cfg.h<=0:raise ValueError('Need at least three positive-duration stages')
    prediction_constraints=None
    if 'prediction_constraints' in spec:
        requested=spec['prediction_constraints']
        if not isinstance(requested,dict) or set(requested)-{'cap_nodes','cap_value'}:
            raise ValueError('Unknown contact prediction constraint option')
        cap_nodes=float(requested.get('cap_nodes',5));cap_value=float(requested.get('cap_value',.005))
        if not np.isfinite([cap_nodes,cap_value]).all() or cap_nodes%1 or not 0<=cap_nodes<=cfg.horizon or cap_value<0:
            raise ValueError('Invalid restitution cap prefix/bound')
        if cap_nodes>0 and cfg.comp_weight<=0:raise ValueError('Restitution cap requires positive contact penalty weight')
        prediction_constraints=dict(cap_nodes=int(cap_nodes),cap_value=cap_value)
    mesh=bool(spec.get('mesh',False));prefix=int(spec.get('prefix',cfg.horizon))
    drop=tuple(spec.get('drop',[]));dedup=bool(spec.get('dedup',False))
    if not set(drop)<=set(['workspace','tilt','upward','paddle_velocity','jerk','endpoint_acceleration']):raise ValueError('Unsupported pruning groups')
    if mesh:steps,times,growth=grid(MeshConfig(**asdict(cfg),mesh_horizon=float(spec.get('mesh_horizon',1.2))))
    else:steps=np.full(cfg.horizon,cfg.h);times=np.arange(cfg.horizon+1)*cfg.h;growth=1.
    # Normalize two tunable residual weights so retuning reuses expensive
    # compiled derivatives. Native assembly scales BOTH residuals and Jacobians.
    factor_cfg=replace(cfg,energy_weight=12.,comp_weight=200.,apex=.75) if spec.get('runtime_cost_scaling') else cfg
    if spec.get('runtime_motion_cost_scaling'):
        factor_cfg=replace(factor_cfg,lateral_weight=8.,ball_velocity_weight=1.,paddle_lateral_weight=1.,pose_weight=2.,velocity_weight=1.,accel_weight=.4,rate_weight=.15,posture_weight=.3)
    if cfg.contact_center_weight:
        factor_cfg=replace(factor_cfg,contact_center_weight=1.)
    parametric=bool(spec.get('runtime_parameters',False))
    radial_scaling=None
    if 'radial_contact_scaling' in spec:
        requested=spec['radial_contact_scaling']
        if not parametric or not isinstance(requested,dict) or set(requested)-{'j0_m_s'}:
            raise ValueError('Radial contact scaling requires runtime parameters and j0_m_s')
        j0=float(requested.get('j0_m_s',0.))
        if not np.isfinite(j0) or not 0<=j0<=8:raise ValueError('Radial contact j0_m_s must be within 0..8')
        radial_scaling=dict(j0_m_s=j0)
    experimental_group_rho=spec.get('experimental_group_rho',False)
    if not isinstance(experimental_group_rho,bool):raise ValueError('Experimental group rho switch must be boolean')
    if experimental_group_rho and (not parametric or 'solver_options' not in spec):
        raise ValueError('Experimental group rho requires runtime parameters and solver_options')
    lift_link_rho=spec.get('residual_lift_link_rho_multiplier')
    if lift_link_rho is not None:
        lift_link_rho=float(lift_link_rho)
        if (not parametric or not spec.get('experimental_residual_lift') or not experimental_group_rho
                or not np.isfinite(lift_link_rho) or not 1<=lift_link_rho<=1000
                or spec.get('solver_options',{}).get('adaptive_rho',0)!=0
                or spec.get('solver_options',{}).get('rho_is_vec',1)!=1
                or spec.get('solver_options',{}).get('contact_rho_multiplier',1)!=1):
            raise ValueError('Lift linkage rho requires the isolated fixed vector-rho residual-lift profile')
    tail_elasticity=None
    if 'tail_endpoint_elasticity' in spec:
        requested=spec['tail_endpoint_elasticity']
        if not parametric or not isinstance(requested,dict) or set(requested)-{'start_s','extra_fraction','weight'}:
            raise ValueError('Tail endpoint elasticity requires parametric factors and start_s/extra_fraction/weight options')
        tail_elasticity=dict(start_s=.30,extra_fraction=.05,weight=10.)
        tail_elasticity.update(requested)
        start,extra,weight=(float(tail_elasticity[k]) for k in ('start_s','extra_fraction','weight'))
        if (not np.isfinite([start,extra,weight]).all() or start<.30 or start>=times[-1]
                or extra<0 or extra>.20 or weight<=0):
            raise ValueError('Invalid bounded tail endpoint elasticity')
        if cfg.actuator_margin+extra>.95+1e-12:
            raise ValueError('Tail endpoint elasticity exceeds 95% research acceleration cap')
        tail_elasticity=dict(start_s=start,extra_fraction=extra,weight=weight)
    arm_interval_model=spec.get('arm_interval_model','legacy')
    if arm_interval_model not in ('legacy','ramp','ramp_prefix'):raise ValueError('Unknown arm interval model')
    ramp=arm_interval_model!='legacy'
    if ramp and not parametric:raise ValueError('Offline ramp variant requires runtime parameters')
    simplification=spec.get('dynamics_simplification')
    if simplification is not None:
        allowed={'neglect_ball_reaction','constant_midpoint_preconditioner'}
        if (not isinstance(simplification,dict) or set(simplification)-allowed
                or any(not isinstance(v,bool) for v in simplification.values()) or ramp):
            raise ValueError('Dynamics simplification requires legacy factors and boolean switches')
        simplification={k:simplification.get(k,False) for k in sorted(allowed)}
        if simplification['constant_midpoint_preconditioner'] and spec.get('solver_options',{}).get('warmstart_arm_shift',0):
            raise ValueError('Constant preconditioner has not been adapted to optional arm-shift repair')
    kinematic=spec.get('kinematic_tail')
    if kinematic is not None:
        if (not parametric or ramp or not isinstance(kinematic,dict)
                or 'dynamic_prefix' not in kinematic or set(kinematic)-{'dynamic_prefix','actuation_proxy_posture','endpoint_accelerations'}):
            raise ValueError('Kinematic tail requires parametric legacy prefix and supported tail settings')
        dynamic_prefix=kinematic['dynamic_prefix']
        if isinstance(dynamic_prefix,bool) or int(dynamic_prefix)!=dynamic_prefix or not 1<=dynamic_prefix<cfg.horizon-1:
            raise ValueError('Kinematic tail needs at least one full dynamic interval plus a boundary and tail')
        dynamic_prefix=int(dynamic_prefix)
        if dynamic_prefix>prefix:raise ValueError('Dynamic prefix must not exceed the fully constrained prefix')
        if 'actuation_proxy_posture' in kinematic:
            posture=np.asarray(kinematic['actuation_proxy_posture'],dtype=float)
            if (posture.shape!=(7,) or not np.isfinite(posture).all()
                    or np.any(posture<robot.q_min) or np.any(posture>robot.q_max)):
                raise ValueError('Tail proxy posture must contain seven finite joint angles within research bounds')
        if (not isinstance(kinematic.get('endpoint_accelerations',False),bool)
                or (kinematic.get('endpoint_accelerations',False) and 'actuation_proxy_posture' not in kinematic)):
            raise ValueError('Endpoint acceleration tail requires a boolean switch and affine actuation proxy')
        if (spec.get('prediction_plane_offset_m',0.)!=0. or spec.get('tail_endpoint_elasticity',{}).get('extra_fraction',0.)!=0.
                or any(spec.get('solver_options',{}).get(k,0.)!=0. for k in ('warmstart_arm_shift','warmstart_repair'))
                or spec.get('bounded_multistart',{}).get('mode',0.)!=0.):
            raise ValueError('Kinematic tail requires zero plane offset, endpoint elasticity, repair and multistart')
    midpoint_posture=None
    if 'constant_midpoint_preconditioner_posture' in spec:
        if (not parametric or ramp or kinematic is not None or simplification is None
                or not simplification['constant_midpoint_preconditioner']):
            raise ValueError('Fixed midpoint posture requires parametric constant-preconditioned legacy dynamics')
        if (any(spec.get('solver_options',{}).get(k,0.)!=0. for k in ('warmstart_arm_shift','warmstart_repair'))
                or spec.get('bounded_multistart',{}).get('mode',0.)!=0.):
            raise ValueError('Fixed midpoint posture does not support optional repair or multistart')
        midpoint_posture=np.asarray(spec['constant_midpoint_preconditioner_posture'],dtype=float)
        if (midpoint_posture.shape!=(7,) or not np.isfinite(midpoint_posture).all()
                or np.any(midpoint_posture<robot.q_min) or np.any(midpoint_posture>robot.q_max)):
            raise ValueError('Fixed midpoint posture must contain seven finite angles inside research joint bounds')
    first_event_center=None
    if 'first_event_center' in spec:
        requested=spec['first_event_center']
        if not parametric or ramp or not isinstance(requested,dict) or set(requested)-{'weight','acquire_lead_s'}:
            raise ValueError('First-event centering requires parametric legacy factors and weight/acquire_lead_s options')
        first_event_center=dict(weight=0.,acquire_lead_s=.15)
        first_event_center.update(requested)
        weight,lead=(float(first_event_center[k]) for k in ('weight','acquire_lead_s'))
        if not np.isfinite([weight,lead]).all() or weight<0 or not .05<=lead<=.5:
            raise ValueError('Invalid first-event centering weight/acquisition lead')
        first_event_center=dict(weight=weight,acquire_lead_s=lead)
    multistart=None
    if 'bounded_multistart' in spec:
        requested=spec['bounded_multistart']
        if (not parametric or ramp or 'solver_options' not in spec
                or not isinstance(requested,dict) or set(requested)-{'mode'}):
            raise ValueError('Bounded multistart requires parametric legacy factors, solver_options and mode only')
        mode=float(requested.get('mode',0.))
        if mode not in (0.,1.,2.):raise ValueError('Bounded multistart mode must be 0, 1 (2+2) or 2 (reset-control4)')
        multistart=dict(mode=mode)
    legacy_edge=spec.get('legacy_contact_prediction',False)
    if not isinstance(legacy_edge,bool):raise ValueError('Legacy contact prediction switch must be boolean')
    if legacy_edge and (not parametric or ramp):raise ValueError('Legacy edge prediction requires parametric legacy arm factors')
    if legacy_edge:
        edge_loss=float(spec.get('prediction_edge_loss',0.))
        if not np.isfinite(edge_loss) or not 0<=edge_loss<=1:raise ValueError('Invalid legacy edge restitution loss')
    legacy_bias=spec.get('legacy_contact_bias',False)
    if not isinstance(legacy_bias,bool):raise ValueError('Legacy contact bias switch must be boolean')
    if legacy_bias and (not parametric or ramp):raise ValueError('Contact bias requires parametric legacy arm factors')
    legacy_plane=spec.get('legacy_contact_plane',False)
    if not isinstance(legacy_plane,bool):raise ValueError('Legacy contact plane switch must be boolean')
    if legacy_plane and (not parametric or ramp):raise ValueError('Contact plane correction requires parametric legacy arm factors')
    if legacy_plane:
        plane_offset=float(spec.get('prediction_plane_offset_m',0.))
        if not np.isfinite(plane_offset) or abs(plane_offset)>.02:raise ValueError('Invalid legacy contact plane offset')
    tuning=ca.MX.sym('tuning',rt.SIZE+(2 if ramp else 0)) if parametric else None
    factor_robot=robot
    if parametric:
        rt.numeric_validation(cfg,spec)
        factor_cfg=cfg
        base=Panda(assembly=spec.get('assembly'),**dict(spec.get('geometry',{}),paddle_normal_rotvec_deg=[0,0,0]))
        factor_robot=rt.Robot(base,tuning[:rt.SIZE])
        factor_cfg=rt.symbolic(cfg,tuning)
    first_builder=tail_builder=create_factors;prediction={}
    if simplification is not None:
        from optimization.simplified_dynamics import create_factors as simplified_factors
        first_builder=tail_builder=partial(simplified_factors,**simplification)
    if ramp:
        from optimization.instant_formulation import create_ramp_factors
        first_builder=create_ramp_factors
        if arm_interval_model=='ramp':tail_builder=create_ramp_factors
        prediction={'prediction_parameters':{'edge_loss':tuning[rt.SIZE],'plane_offset_m':tuning[rt.SIZE+1]}}
    first,full,last,diag,lo,hi,groups=first_builder(factor_cfg,factor_robot,runtime=tuning,**prediction)
    tail_prediction=prediction if tail_builder is first_builder else {}
    _,reduced,_,_,lr,ur,gr=tail_builder(factor_cfg,factor_robot,drop,dedup,'_reduced',runtime=tuning,**tail_prediction)
    _,tail,_,_,lt,ut,gt=tail_builder(factor_cfg,factor_robot,drop,False,'_tail',runtime=tuning,**tail_prediction)
    if kinematic is not None:
        from optimization.kinematic_tail import create_kinematic_tail_factors,acceleration_slot_bounds,integration_metadata
        proxy={} if 'actuation_proxy_posture' not in kinematic else {'actuation_proxy_posture':kinematic['actuation_proxy_posture']}
        if kinematic.get('endpoint_accelerations',False):proxy['endpoint_accelerations']=True
        _,boundary,_,_,lb,ub,gb=create_kinematic_tail_factors(factor_cfg,factor_robot,drop,dedup,'_kinematic_boundary',runtime=tuning,boundary=True,**proxy)
        _,kin_reduced,_,_,lkr,ukr,gkr=create_kinematic_tail_factors(factor_cfg,factor_robot,drop,dedup,'_kinematic_reduced',runtime=tuning,**proxy)
        _,kin_tail,_,_,lkt,ukt,gkt=create_kinematic_tail_factors(factor_cfg,factor_robot,drop,False,'_kinematic_tail',runtime=tuning,**proxy)
        if dynamic_prefix<prefix:
            _,boundary_full,_,_,lbf,ubf,gbf=create_kinematic_tail_factors(factor_cfg,factor_robot,(),False,'_kinematic_boundary_full',runtime=tuning,boundary=True,**proxy)
            _,kin_full,_,_,lkf,ukf,gkf=create_kinematic_tail_factors(factor_cfg,factor_robot,(),False,'_kinematic_full',runtime=tuning,**proxy)
    names=[];offsets=[NP+1];cl=[np.zeros(NP+1)];cu=[np.zeros(NP+1)];ids=[-1]*(NP+1)
    functions=[];counts={};needed={'first':first,'stage':full,'last':last}
    for k in range(cfg.horizon):
        if kinematic is not None and k==dynamic_prefix:
            if k<prefix:f,l,u,g,name=boundary_full,lbf,ubf,gbf,'kinematic_boundary_full'
            else:f,l,u,g,name=boundary,lb,ub,gb,'kinematic_boundary'
        elif kinematic is not None and k>dynamic_prefix:
            if k<prefix:f,l,u,g,name=kin_full,lkf,ukf,gkf,'kinematic_full'
            elif k==cfg.horizon-1 and dedup:f,l,u,g,name=kin_tail,lkt,ukt,gkt,'kinematic_tail'
            else:f,l,u,g,name=kin_reduced,lkr,ukr,gkr,'kinematic_reduced'
        elif k<prefix:f,l,u,g,name=full,lo,hi,groups,'stage'
        elif k==cfg.horizon-1 and dedup:f,l,u,g,name=tail,lt,ut,gt,'tail'
        else:f,l,u,g,name=reduced,lr,ur,gr,'reduced'
        if k==0 and cfg.initial_acceleration_margin>0 and 'endpoint_acceleration' in g:
            if cfg.initial_acceleration_margin>1:raise ValueError('Measured initial acceleration cannot exceed physical bound')
            l=l.copy();u=u.copy();begin=g['endpoint_acceleration'][0]
            l[begin:begin+7]=-cfg.initial_acceleration_margin;u[begin:begin+7]=cfg.initial_acceleration_margin
        needed[name]=f;names.append(name);cl.append(l);cu.append(u);offsets.append(offsets[-1]+len(l))
        for group,(begin,end) in g.items():ids.extend(range(groups[group][0],groups[group][0]+end-begin))
    cl.append([times[-1]]);cu.append([times[-1]]);ids.append(-1)
    for name,f in needed.items():
        nvar=NZ if name in ['first','last'] else 2*NZ;z=ca.MX.sym('z',nvar);p=ca.MX.sym('p',NP)
        extra=[] if tuning is None else [tuning]
        if name=='first':r=ca.MX.zeros(0,1);c=f(z,p,*extra)
        elif name=='last':r,c=f(z,p,*extra)
        else:r,c=f(z[:NZ],z[NZ:],p,*extra)
        r=ca.sqrt(factor_cfg.objective_scale if parametric else cfg.objective_scale)*r
        jr=ca.jacobian(r,z);jc=ca.jacobian(c,z)
        functions.extend([ca.Function(name+'_data',[z,p]+extra,[r,jr,c,jc]),ca.Function(name+'_value',[z,p]+extra,[r,c])])
        counts[name]=dict(variables=nvar,residuals=r.numel(),constraints=c.numel(),jr_nnz=jr.nnz(),jc_nnz=jc.nnz())
    if parametric:
        q=ca.MX.sym('q',7)
        functions.extend([robot.mass,robot.bias,robot.rnea,
          ca.Function('panda_fk',[q,tuning],list(factor_robot.fk(q))),
          ca.Function('panda_contact_fk',[q,tuning],list(factor_robot.contact_fk(q)))])
        # Runtime residuals already carry their actual weights and target apex.
        factor_cfg=cfg
    else:functions.extend([robot.mass,robot.bias,robot.rnea,robot.fk,robot.contact_fk])
    if legacy_edge:
        # A small contact-only correction reuses all expensive legacy factors.
        # Native assembly adds it to restitution row 52 before cost/cap use.
        from optimization.contact_prediction import create_edge_correction
        functions.extend(create_edge_correction(rt.symbolic(cfg,tuning),factor_robot,tuning))
    if legacy_bias:
        from optimization.contact_bias_prediction import create_contact_bias_correction
        functions.extend(create_contact_bias_correction(rt.symbolic(cfg,tuning),factor_robot,tuning))
    if legacy_plane:
        if simplification is None:
            from optimization.contact_plane_prediction import create_contact_plane_correction
            functions.extend(create_contact_plane_correction(rt.symbolic(cfg,tuning),factor_robot,tuning))
        else:
            from optimization.simplified_dynamics import create_contact_plane_correction
            functions.extend(create_contact_plane_correction(rt.symbolic(cfg,tuning),factor_robot,tuning,**simplification))
    cwd=Path.cwd()
    with tempfile.TemporaryDirectory(dir=directory) as temporary:
        try:
            os.chdir(temporary)
            for f in functions:
                f.save(f.name()+'.casadi');cg=ca.CodeGenerator(f.name()+'.c',{'with_header':True});cg.add(f);cg.generate()
        finally:os.chdir(cwd)
        for source in Path(temporary).iterdir():
            target=directory/source.name;data=source.read_bytes()
            if not target.exists() or target.read_bytes()!=data:target.write_bytes(data)
    low,high=bounds(cfg,robot);lower=np.tile(low,(cfg.horizon+1,1));upper=np.tile(high,(cfg.horizon+1,1))
    if kinematic is not None:
        lower[dynamic_prefix+1:,23:30],upper[dynamic_prefix+1:,23:30]=acceleration_slot_bounds(cfg)
    lower[:,32]=upper[:,32]=np.r_[steps,steps[-1]]/SCALE[32];upper[:,33]=times[-1]
    arrays=dict(scale=SCALE,offset=OFFSET,all_lower=lower.ravel(),all_upper=upper.ravel(),
        constraint_lower=np.concatenate(cl),constraint_upper=np.concatenate(cu),stage_steps=steps,stage_times=times,
        home=HOME,origin=robot.origin,torque_limit=TORQUE_LIMIT,velocity_limit=VELOCITY_LIMIT,
        acceleration_limit=ACCEL_LIMIT,torque_rate_limit=TORQUE_RATE_LIMIT,jerk_limit=JERK_LIMIT,
        joint_lower=robot.q_min,joint_upper=robot.q_max)
    from optimization.prediction_fidelity import mask as mask_prediction_fidelity
    fidelity=mask_prediction_fidelity(spec,groups,arrays,offsets,ids,steps,times)
    from optimization.prediction_limits import apply as apply_prediction_limits
    prediction_limits=apply_prediction_limits(spec,groups,arrays,ids)
    from optimization.execution_limits import apply as apply_execution_limits
    execution_limits=apply_execution_limits(spec,arrays)
    if parametric:arrays['factor_parameters']=rt.values(cfg,spec.get('geometry',{}))
    if ramp:
        extra=np.array([spec.get('prediction_edge_loss',0.),spec.get('prediction_plane_offset_m',0.)])
        if not np.isfinite(extra).all() or not 0<=extra[0]<=1 or abs(extra[1])>.02:raise ValueError('Invalid offline contact prediction parameters')
        arrays['factor_parameters']=np.r_[arrays['factor_parameters'],extra]
    ints=dict(stage_offsets=offsets,constraint_ids=ids)
    if 'solver_options' in spec:
        # Full-stage arm checks for the optional warm-start proposal, including
        # constraints pruned from the NLP tail. Use the tightest retained bounds
        # per full row (initial acceleration may have a looser special bound).
        # Profiles without experimental solver options keep their original ABI.
        arm_groups=('position','dynamics','workspace','tilt','upward','paddle_velocity',
                    'acceleration','endpoint_acceleration','torque_rate','jerk')
        arm_ids=[i for name in arm_groups for i in range(*groups[name])]
        arm_low,arm_high=lo.copy(),hi.copy()
        for i in arm_ids:
            rows=np.flatnonzero(np.asarray(ids)==i)
            if len(rows):
                arm_low[i]=np.max(arrays['constraint_lower'][rows])
                arm_high[i]=np.min(arrays['constraint_upper'][rows])
        arrays['seed_arm_lower']=arm_low;arrays['seed_arm_upper']=arm_high
        ints['seed_arm_ids']=arm_ids
    lines=['#pragma once','#include <array>','#include <cmath>','namespace juggling::generated {',
        f'inline constexpr int NX={NX}, NZ={NZ}, NP={NP}, N={cfg.horizon}, CONSTRAINTS={len(ids)};',
        f'#define JUGGLING_BALL_DRAG {cfg.drag!r}',
        f'#define JUGGLING_BALL_GRAVITY_X {cfg.gravity_x!r}',
        f'#define JUGGLING_BALL_GRAVITY_Y {cfg.gravity_y!r}',
        f'#define JUGGLING_BALL_GRAVITY_Z {cfg.gravity_z!r}',
        f'#define JUGGLING_WARMSTART_FEEDBACK_TIME {cfg.warmstart_feedback_time!r}',
        f'#define JUGGLING_FACTOR_APEX {factor_cfg.apex!r}',
        f'#define JUGGLING_ENERGY_MULTIPLIER {(1. if parametric else cfg.energy_weight/factor_cfg.energy_weight)!r}',
        f'#define JUGGLING_CONTACT_MULTIPLIER {(1. if parametric else cfg.comp_weight/factor_cfg.comp_weight)!r}',
        f'#define JUGGLING_CONTACT_CENTER_MULTIPLIER {(1. if parametric else cfg.contact_center_weight)!r}',
        f'inline constexpr bool quadrature={str(mesh).lower()};']
    if simplification is not None:
        lines.append('#define JUGGLING_NEGLECT_BALL_REACTION '+str(int(simplification['neglect_ball_reaction'])))
    if midpoint_posture is not None:
        # Native row mixing reuses the unchanged HOME-preconditioned factors.
        # These fixed references are deliberately outside the runtime arrays;
        # the schema pins them, while the 35-value factor ABI stays unchanged.
        lines.extend(['#define JUGGLING_FIXED_MIDPOINT_POSTURE 1',
            f'inline constexpr int midpoint_dynamics_begin={groups["dynamics"][0]};',
            'inline constexpr std::array<double,7> midpoint_preconditioner_posture={'+', '.join(map(repr,midpoint_posture.tolist()))+'};',
            'inline constexpr std::array<double,7> midpoint_preconditioner_home={'+', '.join(map(repr,HOME.tolist()))+'};'])
    if kinematic is not None:lines.append('#define JUGGLING_KINEMATIC_DYNAMIC_PREFIX '+str(dynamic_prefix))
    if kinematic is not None and kinematic.get('endpoint_accelerations',False):lines.append('#define JUGGLING_KINEMATIC_ENDPOINT_ACCELERATIONS 1')
    if prediction_constraints is not None:
        # Extra native NLP rows reuse residual/Jacobian row 52. Generated
        # factors and their existing row offsets/IDs remain byte-identical.
        lines.extend([f'#define JUGGLING_PREDICTION_CAP_NODES {prediction_constraints["cap_nodes"]}',
            f'inline double prediction_cap_value={prediction_constraints["cap_value"]!r};',
            '#define JUGGLING_PREDICTION_CAP_VALUE juggling::generated::prediction_cap_value'])
    if arm_interval_model=='ramp':
        lines.extend(['#define JUGGLING_RAMP_PREDICTION 1',
            f'#define JUGGLING_PREDICTION_EDGE_LOSS juggling::generated::factor_parameters[{rt.SIZE}]',
            f'#define JUGGLING_PREDICTION_PLANE_OFFSET_M juggling::generated::factor_parameters[{rt.SIZE+1}]'])
    if legacy_edge:
        lines.extend(['#define JUGGLING_LEGACY_EDGE_PREDICTION 1',
            f'inline double prediction_edge_loss={edge_loss!r};',
            '#define JUGGLING_PREDICTION_EDGE_LOSS juggling::generated::prediction_edge_loss'])
    if legacy_bias:lines.append('#define JUGGLING_LEGACY_CONTACT_BIAS 1')
    if legacy_plane:
        lines.extend(['#define JUGGLING_LEGACY_CONTACT_PLANE 1',
            f'inline double prediction_plane_offset_m={plane_offset!r};',
            '#define JUGGLING_PREDICTION_PLANE_OFFSET_M juggling::generated::prediction_plane_offset_m'])
    if experimental_group_rho:lines.append('#define JUGGLING_GROUP_RHO_PROFILE 1')
    if lift_link_rho is not None:
        lines.extend([f'inline double lift_link_rho_multiplier={lift_link_rho!r};',
            '#define JUGGLING_LIFT_LINK_RHO_MULTIPLIER juggling::generated::lift_link_rho_multiplier'])
    if radial_scaling is not None:
        lines.extend([f'inline double radial_contact_j0_m_s={radial_scaling["j0_m_s"]!r};',
            '#define JUGGLING_RADIAL_CONTACT_J0_M_S juggling::generated::radial_contact_j0_m_s'])
    if first_event_center is not None:
        for name,value in first_event_center.items():
            lines.extend([f'inline double first_event_center_{name}={value!r};',
                f'#define JUGGLING_FIRST_EVENT_CENTER_{name.upper()} juggling::generated::first_event_center_{name}'])
    if tail_elasticity is not None:
        for name,value in tail_elasticity.items():
            suffix={'start_s':'START_S','extra_fraction':'EXTRA','weight':'WEIGHT'}[name]
            lines.extend([f'inline double tail_endpoint_{name}={value!r};',
                f'#define JUGGLING_TAIL_ENDPOINT_{suffix} juggling::generated::tail_endpoint_{name}'])
        lines.append(f'inline constexpr int endpoint_acceleration_begin={groups["endpoint_acceleration"][0]};')
    solver_options={}
    if 'solver_options' in spec:
        solver_options=dict(merit_multiplier=1000.,dual_step_blend=0.,adaptive_rho=0.,adaptive_rho_interval=25.,osqp_alpha=1.6,
                            warmstart_repair=0.,qp_retry_warmstart=0.,terminal_position_weight=0.,terminal_velocity_weight=0.,
                            rho_is_vec=1.,scaling=10.,adaptive_rho_tolerance=5.,
                            warmstart_seed_radius=0.,warmstart_seed_lead=.3,warmstart_seed_max_contacts=1.,warmstart_arm_shift=0.,
                            landing_weight=0.,landing_window=.12,contact_rho_multiplier=1.)
        if set(spec['solver_options'])-set(solver_options):raise ValueError('Unknown experimental solver option')
        solver_options.update(spec['solver_options'])
        if not np.isfinite(list(solver_options.values())).all():raise ValueError('Nonfinite experimental solver option')
        if solver_options['merit_multiplier']<=0 or solver_options['dual_step_blend'] not in (0,1) or solver_options['adaptive_rho'] not in (0,1):raise ValueError('Invalid experimental merit/solver option')
        if solver_options['warmstart_repair'] not in (0,1,2,3) or solver_options['qp_retry_warmstart'] not in (0,1,2):raise ValueError('Invalid experimental warmstart switch')
        if min(solver_options['terminal_position_weight'],solver_options['terminal_velocity_weight'])<0:raise ValueError('Negative terminal arm weight')
        if solver_options['rho_is_vec'] not in (0,1) or not 0<=solver_options['scaling']<=20 or solver_options['scaling']%1 or solver_options['adaptive_rho_tolerance']<1:raise ValueError('Invalid OSQP penalty/scaling option')
        seed_radius=solver_options['warmstart_seed_radius']
        if (seed_radius!=0 and not cfg.paddle_radius<seed_radius<=.2) or solver_options['warmstart_seed_lead']<.3:raise ValueError('Invalid virtual tail contact seed geometry')
        if solver_options['warmstart_seed_max_contacts'] not in (1,2,3,4):raise ValueError('Virtual tail contact seed count must be an integer from 1 to 4')
        if solver_options['warmstart_arm_shift'] not in (0,1):raise ValueError('Arm-shift warm start must be 0 or 1')
        if solver_options['warmstart_arm_shift'] and (not parametric or ramp or seed_radius<=0 or prefix<1):raise ValueError('Arm-shift warm start requires parametric legacy factors, full prefix, and virtual seeds')
        if solver_options['landing_weight']<0 or not .02<=solver_options['landing_window']<=.3:raise ValueError('Invalid landing guidance weight/window')
        if solver_options['landing_weight']>0 and not parametric:raise ValueError('Landing guidance study requires runtime parameters')
        if solver_options['adaptive_rho_interval']<1 or solver_options['adaptive_rho_interval']%1 or not 0<solver_options['osqp_alpha']<2:raise ValueError('Invalid OSQP experimental option')
        multiplier=solver_options['contact_rho_multiplier']
        if not 1<=multiplier<=1000:raise ValueError('Contact rho multiplier must be within [1,1000]')
        if multiplier!=1 and (not experimental_group_rho or solver_options['adaptive_rho']!=0 or solver_options['rho_is_vec']!=1):
            raise ValueError('Nondefault contact rho requires experimental_group_rho, adaptive_rho=0 and rho_is_vec=1')
        for name,value in solver_options.items():
            lines.extend([f'inline double solver_{name}={float(value)!r};',f'#define JUGGLING_SOLVER_{name.upper()} juggling::generated::solver_{name}'])
    if multistart is not None:
        mode=multistart['mode']
        if mode:
            if cfg.sqp_iters!=4 or cfg.osqp_iters<1 or cfg.osqp_iters>100 or cfg.osqp_iters%1:
                raise ValueError('Bounded multistart requires 4 SQP iterations and 1..100 OSQP iterations')
            if (solver_options['warmstart_repair']!=0 or solver_options['warmstart_arm_shift']!=0
                    or solver_options['adaptive_rho']!=0 or solver_options['contact_rho_multiplier']!=1
                    or solver_options['warmstart_seed_radius']<=0):
                raise ValueError('Bounded multistart requires virtual seeds, repair/arm_shift/adaptive_rho=0 and contact_rho_multiplier=1')
        lines.extend([f'inline double bounded_multistart_mode={mode!r};',
            '#define JUGGLING_BOUNDED_MULTISTART_MODE juggling::generated::bounded_multistart_mode'])
    for weight in ['lateral_weight','ball_velocity_weight','paddle_lateral_weight','pose_weight','velocity_weight','accel_weight','rate_weight','posture_weight']:
        lines.append(f'#define JUGGLING_{weight.upper()}_MULTIPLIER {(1. if parametric else getattr(cfg,weight)/getattr(factor_cfg,weight))!r}')
    for name in ['position_weight','velocity_weight']:
        gain=float(spec.get('nullspace_regularization',{}).get(name,0.))
        if not np.isfinite(gain) or not 0 <= gain <= 10:raise ValueError('Nullspace weights must be finite and within [0,10]')
        lines.append(f'#define JUGGLING_NULLSPACE_{name.upper()} {gain!r}')
    servo_jerk=float(spec.get('execution',{}).get('jerk_fraction',.2))
    if not 0 < servo_jerk <= .2:raise ValueError('Execution jerk fraction must be within (0, 0.2]')
    lines.append(f'#define JUGGLING_SERVO_JERK_FRACTION {servo_jerk!r}')
    for name,default in [('position_gain',100.),('velocity_gain',20.)]:
        gain=float(spec.get('execution',{}).get(name,default))
        if not 0 < gain <= default:raise ValueError('Servo gain must be positive and no larger than the original gain')
        lines.append(f'#define JUGGLING_SERVO_{name.upper()} {gain!r}')
    for name,value in asdict(cfg).items():
        if isinstance(value,(int,float)) and not isinstance(value,bool):lines.append(f'inline constexpr double {name}={value!r};')
    for name,values in arrays.items():
        def number(v):return 'INFINITY' if np.isposinf(v) else '-INFINITY' if np.isneginf(v) else repr(float(v))
        lines.append(f'inline const std::array<double,{len(values)}> {name}={{{", ".join(map(number,values))}}};')
    for name,values in ints.items():lines.append(f'inline const std::array<int,{len(values)}> {name}={{{", ".join(map(str,values))}}};')
    for suffix in ['data','value']:
        quoted=', '.join(json.dumps(n+'_'+suffix) for n in names)
        lines.append(f'inline const std::array<const char*,{len(names)}> stage_{suffix}_names={{{quoted}}};')
    if 'solver_options' in spec:lines.append(f'inline constexpr int seed_dynamics_begin={groups["dynamics"][0]};')
    lines.extend([f'inline constexpr int contact_begin={groups["contact"][0]}, contact_end={groups["contact"][1]};',
        f'inline constexpr const char * mass_name="{robot.mass.name()}", *bias_name="{robot.bias.name()}", *rnea_name="{robot.rnea.name()}", *fk_name="{robot.fk.name()}", *contact_fk_name="{robot.contact_fk.name()}";','}'])
    if parametric:
        scalars={n:float(v) for n,v in asdict(cfg).items() if isinstance(v,(int,float)) and not isinstance(v,bool)}
        scalars.update({'solver_'+n:float(v) for n,v in solver_options.items()})
        if lift_link_rho is not None:scalars['lift_link_rho_multiplier']=lift_link_rho
        if prediction_constraints is not None:scalars['prediction_cap_value']=prediction_constraints['cap_value']
        if legacy_edge:scalars['prediction_edge_loss']=edge_loss
        if legacy_plane:scalars['prediction_plane_offset_m']=plane_offset
        if radial_scaling is not None:scalars['radial_contact_j0_m_s']=radial_scaling['j0_m_s']
        if multistart is not None:scalars['bounded_multistart_mode']=multistart['mode']
        if first_event_center is not None:
            scalars.update({'first_event_center_'+name:value for name,value in first_event_center.items()})
        if tail_elasticity is not None:
            scalars.update({'tail_endpoint_'+name:value for name,value in tail_elasticity.items()})
        # Keep compile-time dimensions immutable; all scalar tuning and numeric
        # bounds/grid values load once, including in the robot child process.
        macros={'BALL_DRAG':'drag','BALL_GRAVITY_X':'gravity_x','BALL_GRAVITY_Y':'gravity_y','BALL_GRAVITY_Z':'gravity_z','WARMSTART_FEEDBACK_TIME':'warmstart_feedback_time','FACTOR_APEX':'apex'}
        for name in ['position_weight','velocity_weight']:
            key='nullspace_'+name;scalars[key]=float(spec.get('nullspace_regularization',{}).get(name,0.));macros['NULLSPACE_'+name.upper()]=key
        for name,default in [('position_gain',100.),('velocity_gain',20.),('jerk_fraction',.2)]:
            key='servo_'+name;scalars[key]=float(spec.get('execution',{}).get(name,default));macros['SERVO_'+name.upper()]=key
        for key in list(scalars):
            if key.startswith(('nullspace_','servo_')):lines.insert(-1,f'inline double {key}={scalars[key]!r};')
        lines=[line.replace('inline constexpr double','inline double').replace('inline const std::array<double','inline std::array<double') for line in lines]
        for i,line in enumerate(lines):
            for macro,name in macros.items():
                if line.startswith('#define JUGGLING_'+macro+' '):lines[i]='#define JUGGLING_'+macro+' juggling::generated::'+name
        structure=dict(version=1,N=cfg.horizon,prefix=prefix,drop=drop,dedup=dedup,mesh=mesh,
          assembly=spec.get('assembly'),geometry={k:v for k,v in spec.get('geometry',{}).items() if k!='paddle_normal_rotvec_deg'},
          counts=counts,ints=ints,stage_names=names)
        if ramp:structure['arm_interval_model']=arm_interval_model
        if simplification is not None:structure['dynamics_simplification']=simplification
        if midpoint_posture is not None:structure['constant_midpoint_preconditioner_posture']=midpoint_posture.tolist()
        if kinematic is not None:structure['kinematic_tail']=kinematic
        if legacy_edge:structure['legacy_contact_prediction']=True
        if legacy_bias:structure['legacy_contact_bias']=True
        if legacy_plane:structure['legacy_contact_plane']=True
        if radial_scaling is not None:structure['radial_contact_scaling']=True
        if multistart is not None:structure['bounded_multistart']=True
        if first_event_center is not None:structure['first_event_center']=True
        if tail_elasticity is not None:structure['tail_endpoint_elasticity']=True
        if experimental_group_rho:structure['experimental_group_rho']=True
        if lift_link_rho is not None:structure['residual_lift_link_rho_multiplier']=True
        if prediction_constraints is not None:
            structure['prediction_constraints']=dict(cap_nodes=prediction_constraints['cap_nodes'])
        loader,schema=rt.write_runtime(directory,scalars,arrays,structure)
        lines[1:1]=['#include <fstream>','#include <cstdlib>','#include <string>','#include <stdexcept>','#define JUGGLING_PARAMETRIC_FACTORS 1']
        lines[-1:-1]=loader
    (directory/'model_config.hpp').write_text('\n'.join(lines)+'\n')
    metadata=dict(runtime_schema=schema if parametric else None, runtime_parameters=list(rt.FIELDS) if parametric else [],config=asdict(cfg),spec=spec,mesh_growth=growth,stage_steps=steps.tolist(),stage_times=times.tolist(),
        groups=groups,stage_offsets=offsets,stage_names=names,constraint_ids=ids,counts=counts,casadi_version=ca.__version__,
        functions=[f.name() for f in functions],quadrature='running sqrt(h_k/.03), torque difference additionally .03/h_k; event penalties and terminal unchanged' if mesh else 'baseline',
        arrays={n:v.tolist() for n,v in arrays.items()})
    if simplification is not None:
        metadata['dynamics_simplification']=dict(simplification,
            midpoint_preconditioner_reference='inverse mass at HOME, including configured paddle payload',
            endpoint_acceleration_unchanged=True,physical_plant_changed=False)
    if midpoint_posture is not None:
        metadata['midpoint_preconditioner_transform']=dict(reference_posture_rad=midpoint_posture.tolist(),
            factor_reference='HOME, unchanged generated factors',
            native_transform='M(reference)^-1 M(HOME), applied to dynamics values and Jacobians after corrections',
            fixed_per_controller=True,endpoint_acceleration_unchanged=True,physical_plant_changed=False)
    if kinematic is not None:metadata['kinematic_tail']=integration_metadata(dynamic_prefix,**proxy)
    if spec.get('experimental_residual_lift'):metadata['experimental_residual_lift']=True
    if lift_link_rho is not None:
        metadata['residual_lift_link_rho']=dict(multiplier=lift_link_rho,
            scope='Only auxiliary residual equality penalties in scaled OSQP coordinates',
            exact_QP_and_merit_unchanged=True,physical_truth_changed=False)
    if prediction_constraints is not None:
        metadata['prediction_constraints']=dict(prediction_constraints,row_begin=len(ids),
            row_end=len(ids)+prediction_constraints['cap_nodes'],
            expression='(j/8)*(normal restitution residual/8) <= cap_value',
            source_residual_row=52,dual_shift='cleared with contact multipliers')
        metadata['native_nonlinear_rows']=len(ids)+prediction_constraints['cap_nodes']
    if multistart is not None:
        metadata['bounded_multistart']=dict(multistart,
            modes={'0':'off; legacy path','1':'successful first-virtual seed: two SQP per seed; exact final merit',
                   '2':'same trigger: regenerated seed only, fresh workspace, four SQP'},
            total_sqp_iterations=4,osqp_iterations_per_qp=int(cfg.osqp_iters),
            workspaces='symmetric fresh public OSQP setup on triggered steps; winner state retained',
            trigger='successful regeneration; first virtual contact has chronological contact_ordinal=1',
            selection='same full nonlinear cost plus configured merit multiplier times objective_scale times L1 violation',
            runtime_abi='one additional scalar outside the unchanged 19 solver options; absent adds none',
            trace_environment='JUGGLING_MULTISTART_TRACE',physical_truth_changed=False)
    if fidelity is not None:metadata['prediction_fidelity']=fidelity
    if prediction_limits is not None:metadata['prediction_limits']=prediction_limits
    if execution_limits is not None:metadata['simulation_execution_limits']=execution_limits
    if radial_scaling is not None:
        metadata['radial_contact_scaling']=dict(radial_scaling,
            original_constraint_id=groups['contact'][0]+2,
            rows=[row for row,i in enumerate(ids) if i==groups['contact'][0]+2],
            physical_impulse_column=30,normalized_impulse_scale=float(SCALE[30]),
            expression='c_new=c_old*hypot(8,j0_m_s)/hypot(j,j0_m_s); j0_m_s=0 bypasses exactly',
            zero_bound_feasible_set_unchanged=True,physical_truth_changed=False,
            diagnostic_note='Native radial violation is scaled; divide by the positive multiplier to recover the original row.')
    if first_event_center is not None:
        metadata['first_event_center']=dict(first_event_center,
            cost_scope='One frozen next-event time; exact arm interpolation; no impulse or contact-fraction dependence',
            penalty='Tangential observed-flight target to preferred contact center; discrete event cost without dt scaling',
            physical_constraints_changed=False,physical_truth_changed=False)
    if tail_elasticity is not None:
        metadata['tail_endpoint_elasticity']=dict(tail_elasticity,
            bound_scope='Existing tail endpoint-acceleration rows only; boundary node and prefix protected',
            penalty='One squared hinge per endpoint/joint to original bounds; trapezoidal nodal dt/.03; fixed .05 normalization',
            hard_cap_research_fraction=.95,physical_truth_changed=False)
    if experimental_group_rho:
        cap_count=prediction_constraints['cap_nodes'] if prediction_constraints else 0
        metadata['experimental_group_rho']=dict(
            multiplier=solver_options['contact_rho_multiplier'],
            radial_rows=[row for row,i in enumerate(ids) if i==groups['contact'][0]+2],
            restitution_cap_rows=list(range(len(ids),len(ids)+cap_count)),
            policy='Multiply only selected finite inequality penalties in scaled OSQP; preserve equalities/free/other rows',
            osqp_source_commit='1572ae068e9ce9ca723cf8223548ade1ff7acc29',
            cmake_option='JUGGLING_EXPERIMENTAL_GROUP_RHO',
            default='Multiplier 1 bypasses the bridge entirely',
            runtime_abi='19 solver scalars; previous source_arm had 18; no solver_options means no extra scalar')
    (directory/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n');print(json.dumps(dict(directory=str(directory),stages=cfg.horizon,rows=len(ids),counts=counts),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('spec',type=Path);p.add_argument('output',type=Path);a=p.parse_args();export(json.loads(a.spec.read_text()),a.output)
