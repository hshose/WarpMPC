"""Export a recorded Newton measurement/actuation sequence for C++-only timing."""
import argparse
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    control=np.load(a.run/'arrays.npz');plant=np.load(a.run/'plant.npz')
    dt=float(plant['t'][1]-plant['t'][0]);index=np.rint(control['t'][:-1]/dt).astype(int)
    # A plant sample stores the ramp midpoint torque. Recover the torque at
    # the exact measurement/solve time, including any contact endpoint holds.
    applied=plant['tau'][index]-.5*dt*plant['torque_rate'][index]
    a.output.parent.mkdir(parents=True,exist_ok=True)
    np.savetxt(a.output,np.c_[control['t'][:-1],control['x'][:-1],applied],delimiter=',',header='time, state23, applied_total_torque7')


if __name__=='__main__':main()
