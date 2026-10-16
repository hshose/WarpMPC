"""Check two SQP passes equal two unshifted one-pass updates of the same NLP."""
import argparse,json
from pathlib import Path
import numpy as np
from cpp.python_controller import NativeController
from arm.model import Panda

def main():
 p=argparse.ArgumentParser();p.add_argument('single',type=Path);p.add_argument('double',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 meta=json.loads((a.single/'metadata.json').read_text());spec=meta['spec'];robot=Panda(assembly=spec['assembly'],**spec['geometry'])
 recording=Path('runs/hardware_moving_mpc_20261005/moving_mpc_endurance_20261005_170049');home=json.loads((recording/'fixture_home.json').read_text());rows=[json.loads(l) for l in (recording/'mpc.jsonl').read_text().splitlines()]
 one=NativeController(library=a.single.parent/'controller/libjuggling_controller.so',factors=a.single);two=NativeController(library=a.double.parent/'controller/libjuggling_controller.so',factors=a.double)
 for c in [one,two]:c.set_reference(home['face_offset_m'],home['q_home'])
 x=np.array(rows[0]['state']);tau=np.asarray(robot.bias(x[6:13],x[13:20])).ravel()
 for c in [one,two]:c.initialize(x,tau)
 errors=[]
 for row in rows[:10]:
  x=np.array(row['state']);tau=np.asarray(robot.bias(x[6:13],x[13:20])).ravel()
  one.step(x,tau,0.);Y1,_,s1=one.step(x,tau,0.);Y2,_,s2=two.step(x,tau,0.)
  error=float(np.max(abs(Y1-Y2)));errors.append(error)
  assert error<1e-10,(error,s1,s2)
  assert int(s1[17])==1 and int(s2[17])==2
  assert int(s2[12])==2*int(s1[12])==100
  assert abs(s1[5]-s2[5])<1e-10
 result=dict(passed=True,plan_errors=errors,sqp_passes=int(s2[17]),total_osqp_iterations=int(s2[12]))
 a.output.write_text(json.dumps(result,indent=2)+'\n');print(a.output.read_text());one.close();two.close()
if __name__=='__main__':main()
