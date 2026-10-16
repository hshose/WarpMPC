import bootstrap
import argparse,json,time
from pathlib import Path
import numpy as np
from factors import Factors
from warmstart import WarmStart
p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('--backend',default='warp');a=p.parse_args()
f=Factors(a.directory/'gpu_factors');fixture=np.load(a.directory/'native_fixtures.npz');w=WarmStart(f,a.backend)
t=time.monotonic();z,y=w.initialize(np.asarray(fixture['p'][:,:23],np.float32),np.asarray(fixture['p'][:,23:30],np.float32))
error=float(np.max(abs(np.asarray(z)-fixture['z'])));report=dict(max_scaled_seed_error=error,passed=error<.002,elapsed_s=time.monotonic()-t)
print(json.dumps(report,indent=2));(a.directory/'warmstart_parity.json').write_text(json.dumps(report,indent=2)+'\n');assert report['passed']
