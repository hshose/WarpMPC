"""Compare plans in native variable scales and in executed acceleration units."""
import casadi as ca
import numpy as np

def metrics(factors,actual,expected):
    actual=np.asarray(actual);expected=np.asarray(expected);error=actual-expected;arr=factors.arrays
    mass=ca.Function.load(str(factors.directory/'panda_mass.casadi'));bias=ca.Function.load(str(factors.directory/'panda_bias.casadi'))
    def accelerations(plan):
        return np.array([[np.linalg.solve(np.asarray(mass(row[6:13])),row[23:30]-np.asarray(bias(row[6:13],row[13:20])).ravel()) for row in episode[:4]] for episode in plan])
    acceleration_error=accelerations(actual)-accelerations(expected)
    scaled=float(np.max(abs(error[:,:4]/np.asarray(arr['scale']))));acc=float(np.max(abs(acceleration_error)/np.asarray(arr['acceleration_limit'])))
    return dict(passed=scaled<.003 and acc<.01,max_plan_abs=float(np.max(abs(error))),max_first4_scaled_error=scaled,
        max_first4_joint_abs=float(np.max(abs(error[:,:4,6:20]))),max_first4_acceleration_limit_fraction=acc,
        tolerances=dict(first4_scaled=.003,acceleration_limit_fraction=.01))
