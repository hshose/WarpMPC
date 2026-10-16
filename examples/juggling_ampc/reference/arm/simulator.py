"""0.5 ms Panda forward dynamics and independently detected finite-mass impacts."""
from dataclasses import dataclass
import numpy as np
import pinocchio as pin
from .model import Panda

@dataclass
class SimConfig:
    ball_mass:float=.0027
    radius:float=.02
    paddle_radius:float=.075
    restitution:float=.86
    drag:float=.02
    gravity_x:float=0.
    gravity_y:float=0.
    gravity_z:float=-9.81
    damping:float=0.
    max_interval:float=.0005

class Simulator:
    def __init__(self,robot=None,cfg=None):
        self.robot=Panda() if robot is None else robot
        self.cfg=SimConfig() if cfg is None else cfg
        self.data=self.robot.pin.createData()

    def arm_advance(self,q,v,ta,tb,h):
        model=self.robot.pin;data=self.data;damping=self.cfg.damping
        def rhs(q,v,tau):
            return pin.aba(model,data,q,v,tau-damping*v).copy()
        tm=.5*(ta+tb)
        a1=rhs(q,v,ta);v2=v+.5*h*a1;q2=q+.5*h*v
        a2=rhs(q2,v2,tm);v3=v+.5*h*a2;q3=q+.5*h*v2
        a3=rhs(q3,v3,tm);v4=v+h*a3;q4=q+h*v3
        a4=rhs(q4,v4,tb)
        return q+h/6*(v+2*v2+2*v3+v4),v+h/6*(a1+2*a2+2*a3+a4)

    def free(self,x,ta,tb,h,wind,acc=None):
        out=x.copy();drag=self.cfg.drag;force=wind+np.array([self.cfg.gravity_x,self.cfg.gravity_y,self.cfg.gravity_z])
        if drag:
            decay=np.exp(-drag*h);phi1=-np.expm1(-drag*h)/drag;phi2=(h-phi1)/drag
            out[:3]=x[:3]+phi1*x[3:6]+phi2*force;out[3:6]=decay*x[3:6]+phi1*force
        else:
            out[:3]=x[:3]+h*x[3:6]+.5*h*h*force;out[3:6]=x[3:6]+h*force
        if acc is None:out[6:13],out[13:20]=self.arm_advance(x[6:13],x[13:20],ta,tb,h)
        else:
            out[6:13]=x[6:13]+h*x[13:20]+.5*h*h*acc;out[13:20]=x[13:20]+h*acc
        return out

    def geometry(self,x):
        pos,n,J,N=self.robot.numeric_fk(x[6:13]);pos-=self.robot.origin
        rel=x[:3]-pos;gap=n@rel-self.cfg.radius
        tangent=rel-(n@rel)*n;contactJ=n@J-rel@N
        speed=n@x[3:6]-contactJ@x[13:20]
        return gap,np.linalg.norm(tangent),n,contactJ,speed

    def step(self,x,ta,tb,h,wind=None,acc=None):
        wind=np.zeros(3) if wind is None else wind
        cfg=self.cfg;contacts=[];state=x.copy();elapsed=0.
        # Bounded search intervals also cover longer nominal warmstart steps.
        while elapsed<h-1e-14:
            dt=min(cfg.max_interval,h-elapsed)
            ua=ta+(tb-ta)*(elapsed/h);ub=ta+(tb-ta)*((elapsed+dt)/h)
            end=self.free(state,ua,ub,dt,wind,acc)
            ga=self.geometry(state);gb=self.geometry(end)
            if ga[0]>=-1e-10 and gb[0]<=0 and (ga[4]<0 or gb[4]<0):
                lo=0.;hi=dt
                while hi-lo>1e-10:
                    mid=.5*(lo+hi);um=ua+(ub-ua)*(mid/dt)
                    trial=self.free(state,ua,um,mid,wind,acc)
                    if self.geometry(trial)[0]>0:lo=mid
                    else:hi=mid
                hit=.5*(lo+hi);um=ua+(ub-ua)*(hit/dt)
                collision=self.free(state,ua,um,hit,wind,acc)
                gap,rad,n,J,speed=self.geometry(collision)
                if rad<=cfg.paddle_radius and speed<0:
                    M=pin.crba(self.robot.pin,self.data,collision[6:13]).copy()
                    response=np.linalg.solve(M,J)
                    impulse=-(1+cfg.restitution)*speed/(1/cfg.ball_mass+J@response)
                    collision[3:6]+=impulse/cfg.ball_mass*n
                    collision[13:20]-=impulse*response
                    collision[:3]+=(1e-10-gap)*n
                    elapsed+=hit
                    contacts.append(dict(time=elapsed,position=(collision[:3]-cfg.radius*n).tolist(),normal=n.tolist(),
                        incoming_normal_speed=float(speed),impulse_Ns=float(impulse),radial_distance=float(rad),
                        ball_velocity_after=collision[3:6].tolist(),joint_velocity_jump=(-impulse*response).tolist()))
                    state=collision
                    if len(contacts)>16:raise RuntimeError('Unsupported resting contact')
                    continue
            state=end;elapsed+=dt
        return state,contacts
