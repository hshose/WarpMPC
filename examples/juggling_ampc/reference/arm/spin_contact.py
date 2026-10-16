"""Independent NumPy finite-arm hollow-ball frictional impact calculation."""
import numpy as np
from scipy.optimize import brentq


def impact(normal,tangents,J,mass_matrix,relative,ball_mass=.0027,radius=.02,
           restitution=.88,friction=.3,inertia_ratio=2/3):
    n=np.asarray(normal);T=np.asarray(tangents)
    K=np.eye(3)+(np.eye(3)-np.outer(n,n))/inertia_ratio+ball_mass*J@np.linalg.solve(mass_matrix,J.T)
    vn=n@relative
    def impulse(j):
        desired=-np.linalg.solve(T.T@K@T,T.T@(relative+K@n*j))
        cap=friction*j
        jt=desired*cap/np.sqrt(cap*cap+desired@desired+1e-10)
        return j*n+T@jt
    j=brentq(lambda j:n@K@impulse(j)+(1+restitution)*vn,0.,max(1.,-3*vn))
    u=impulse(j)
    return u,-np.cross(n,u)/(inertia_ratio*radius),-ball_mass*np.linalg.solve(mass_matrix,J.T@u),j
