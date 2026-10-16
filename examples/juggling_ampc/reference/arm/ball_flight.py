"""Exact constant-gravity, isotropic linear-drag flight used by MPC factors."""
import casadi as ca

def flight(position, velocity, dt, drag, gravity=(0.,0.,-9.81)):
    gravity = ca.vertcat(*gravity)
    if isinstance(drag,(ca.MX,ca.SX)):
        # Analytic zero-drag limit; safe denominator even in the inactive branch.
        x=drag*dt;d=ca.fmax(drag,1e-8)
        vd=ca.if_else(ca.fabs(x)<1e-5,dt*(1-x/2+x*x/6-x*x*x/24),-ca.expm1(-x)/d)
        pd=ca.if_else(ca.fabs(x)<1e-5,dt*dt*(.5-x/6+x*x/24-x*x*x/120),(dt-vd)/d)
        return position+vd*velocity+pd*gravity,ca.exp(-x)*velocity+vd*gravity
    if drag == 0:
        return position+dt*velocity+.5*dt*dt*gravity, velocity+dt*gravity
    decay = ca.exp(-drag*dt)
    vd = -ca.expm1(-drag*dt)/drag
    pd = (dt-vd)/drag
    return position+vd*velocity+pd*gravity, decay*velocity+vd*gravity
