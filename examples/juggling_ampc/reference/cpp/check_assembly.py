"""Compare the hardware assembly dynamics against independent Pinocchio RNEA."""
import argparse
import json
from pathlib import Path

import numpy as np
import pinocchio as pin
from arm.model import Panda,HOME,TORQUE_LIMIT,PADDLE_OFFSET


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('spec',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--homing-trace',type=Path)
    args=parser.parse_args()
    spec=json.loads(args.spec.read_text());assembly=spec['assembly']
    original=Panda();robot=Panda(assembly=assembly,**spec.get('geometry',{}))
    rng=np.random.default_rng(20260930)
    errors=dict(mass_matrix=0.,inverse_dynamics=0.,paddle_position=0.,paddle_normal=0.)
    for k in range(100):
        q=HOME if k==0 else rng.uniform(robot.q_min,robot.q_max)
        v=rng.uniform(-.3,.3,7);a=rng.uniform(-1.,1.,7)
        M=np.asarray(robot.mass(q));Mpin=pin.crba(robot.pin,robot.data,q)
        tau=np.asarray(robot.rnea(q,v,a)).ravel();taupin=pin.rnea(robot.pin,robot.data,q,v,a)
        errors['mass_matrix']=max(errors['mass_matrix'],float(abs(M-Mpin).max()))
        errors['inverse_dynamics']=max(errors['inverse_dynamics'],float(abs(tau-taupin).max()))
        current=robot.fk(q);original.numeric_fk(q)
        flange=original.data.oMf[original.pin.getFrameId('panda_link8')]
        expected_position=flange.translation+flange.rotation@robot.assembly_rotation@(PADDLE_OFFSET-[0,0,robot.paddle_surface_offset])
        expected_normal=-flange.rotation[:,2]
        errors['paddle_position']=max(errors['paddle_position'],float(abs(np.asarray(current[0]).ravel()-expected_position).max()))
        errors['paddle_normal']=max(errors['paddle_normal'],float(abs(np.asarray(current[1]).ravel()-expected_normal).max()))
        assert np.linalg.eigvalsh(M).min()>0
    mass_change=pin.computeTotalMass(robot.pin)-pin.computeTotalMass(original.pin)
    assert abs(mass_change-(assembly['mass_kg']-.25))<1e-12
    assert max(errors.values())<1e-10,errors
    result=dict(passed=True,sampled_states=100,physical_motion=False,errors=errors,
                original_assembly_mass_kg=.25,new_assembly_mass_kg=assembly['mass_kg'],
                total_robot_mass_change_kg=mass_change,
                home_gravity_change_Nm=(np.asarray(robot.bias(HOME,np.zeros(7)))-np.asarray(original.bias(HOME,np.zeros(7)))).ravel().tolist(),
                mounting_geometry=spec.get('geometry',{}),assembly_com_flange_m=robot.assembly_com_flange.tolist(),
                assembly_inertia_flange_kg_m2=robot.assembly_inertia_flange.tolist(),
                home_face_center_base_m=robot.origin.tolist(),
                notes='Whole assembly replaces both old rigid bodies; mounting rotates mesh/COM/inertia consistently. Default model/URDF unchanged.')
    if args.homing_trace:
        trace=np.load(args.homing_trace)['observed_robot'][::50]
        tau=np.array([np.asarray(robot.rnea(row[1:8],row[8:15],row[15:22])).ravel() for row in trace])
        result['homing_prediction']=dict(samples=len(trace),trace=str(args.homing_trace),
            max_abs_torque_Nm=abs(tau).max(axis=0).tolist(),max_torque_ratio=float((abs(tau)/TORQUE_LIMIT).max()),
            within_research_torque_bounds=bool((abs(tau)<=TORQUE_LIMIT).all()),
            note='CAD assembly on the existing Panda identified dynamics; not SDK torque validation')
        assert result['homing_prediction']['within_research_torque_bounds']
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
