"""Panda URDF, differentiable Newton–Euler dynamics, and independent Pinocchio model."""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import casadi as ca
import pinocchio as pin
import yaml
from scipy.spatial.transform import Rotation

MODEL_DIR=Path(__file__).resolve().parents[1]/'models'/'panda'
HOME=np.array([0.,-np.pi/4,0.,-3*np.pi/4,0.,np.pi/2,np.pi/4])
PADDLE_OFFSET=np.array([.16,0.,.04])
PADDLE_SURFACE_OFFSET=.006  # Top face of the 12 mm-thick disk, along its normal.
TORQUE_LIMIT=np.array([60.,60.,60.,60.,8.,8.,8.])
VELOCITY_LIMIT=np.array([1.5]*4+[1.8]*3)
ACCEL_LIMIT=np.array([10.,5.,7.,8.,10.,13.,13.])
TORQUE_RATE_LIMIT=np.full(7,300.)
JERK_LIMIT=np.array([1000.,500.,650.,800.,1000.,1300.,1300.])
# Stage 3 collision-checked printed bracket: upright at the flange and a beam
# behind the racket face. The old box crossed the usable hitting region.
FIXTURE_PARTS=[(.02,np.array([0.,0.,.0325]),np.array([.03,.04,.065])),
               (.05,np.array([.06,0.,.065]),np.array([.12,.04,.03]))]


def generate_urdf():
    """Mesh-free URDF from Franka's Panda geometry and Gaz identified inertias."""
    inertias=yaml.safe_load((MODEL_DIR/'inertial.yaml').read_text())
    limits=yaml.safe_load((MODEL_DIR/'joint_limits.yaml').read_text())
    root=ET.Element('robot',name='panda_juggling')
    root.append(ET.Comment('Derived from Franka franka_ros, Apache-2.0; see models/panda/README.md. No gripper.'))
    origins=[([0,0,.333],[0,0,0]),([0,0,0],[-np.pi/2,0,0]),
             ([0,-.316,0],[np.pi/2,0,0]),([.0825,0,0],[np.pi/2,0,0]),
             ([-.0825,.384,0],[-np.pi/2,0,0]),([0,0,0],[np.pi/2,0,0]),
             ([.088,0,0],[np.pi/2,0,0]),([0,0,.107],[0,0,0]),
             (PADDLE_OFFSET,[np.pi,0,0])]
    def nums(v):return ' '.join(f'{float(x):.16g}' for x in v)
    ET.SubElement(root,'link',name='panda_link0')
    for k,(xyz,rpy) in enumerate(origins,1):
        name=f'panda_link{k}' if k<9 else 'paddle'
        link=ET.SubElement(root,'link',name=name)
        if k<=7:
            source=inertias[f'link{k}'];mass=source['mass'];com=source['origin']['xyz'];I=source['inertia']
        elif k==8: # 70 g bracket, two uniform boxes with parallel-axis inertia.
            mass=sum(v[0] for v in FIXTURE_PARTS);center=sum(m*c for m,c,d in FIXTURE_PARTS)/mass;com=nums(center)
            tensor=np.zeros((3,3))
            for m,c,d in FIXTURE_PARTS:
                r=c-center;tensor+=np.diag(m/12*(sum(d*d)-d*d))+m*((r@r)*np.eye(3)-np.outer(r,r))
            I={a+b:tensor[i,j] for i,a in enumerate('xyz') for j,b in enumerate('xyz') if i<=j}
        else: # 180 g racket head, 75 mm radius, 12 mm thickness, uniform disk.
            mass=.18;com='0 0 0';rr=.075;th=.012
            I=dict(xx=mass*(3*rr*rr+th*th)/12,yy=mass*(3*rr*rr+th*th)/12,zz=mass*rr*rr/2,xy=0,xz=0,yz=0)
        ine=ET.SubElement(link,'inertial')
        ET.SubElement(ine,'origin',xyz=com,rpy='0 0 0')
        ET.SubElement(ine,'mass',value=str(mass))
        ET.SubElement(ine,'inertia',**{'i'+n:str(float(v)) for n,v in I.items()})
        joint=ET.SubElement(root,'joint',name=f'panda_joint{k}',type='revolute' if k<=7 else 'fixed')
        ET.SubElement(joint,'parent',link=f'panda_link{k-1}')
        ET.SubElement(joint,'child',link=name)
        ET.SubElement(joint,'origin',xyz=nums(xyz),rpy=nums(rpy))
        if k<=7:
            ET.SubElement(joint,'axis',xyz='0 0 1')
            ET.SubElement(joint,'limit',**{n:str(v) for n,v in limits[f'joint{k}']['limit'].items()})
    ET.indent(root)
    path=MODEL_DIR/'panda_juggling.urdf'
    ET.ElementTree(root).write(path,encoding='unicode',xml_declaration=True)
    return path


def rz(a):
    return ca.vertcat(ca.horzcat(ca.cos(a),-ca.sin(a),0),ca.horzcat(ca.sin(a),ca.cos(a),0),ca.horzcat(0,0,1))


class Panda:
    def __init__(self,assembly=None,assembly_yaw_deg=0.,paddle_surface_offset=PADDLE_SURFACE_OFFSET,paddle_normal_rotvec_deg=(0.,0.,0.)):
        path=MODEL_DIR/'panda_juggling.urdf'
        if not path.exists():generate_urdf()
        root=ET.parse(path).getroot();links={v.get('name'):v for v in root.findall('link')}
        if not np.isfinite(assembly_yaw_deg) or not np.isfinite(paddle_surface_offset) or not 0<paddle_surface_offset<.05:
            raise ValueError('Invalid attachment geometry')
        if assembly is None and assembly_yaw_deg!=0:
            raise ValueError('A mounting rotation requires complete assembly mass properties')
        correction=np.asarray(paddle_normal_rotvec_deg,dtype=float)
        if correction.shape!=(3,) or not np.isfinite(correction).all() or np.linalg.norm(correction)>10:
            raise ValueError('Paddle normal correction must be a finite rotation vector within 10 degrees')
        self.paddle_normal_rotation=Rotation.from_rotvec(np.deg2rad(correction)).as_matrix()
        self.paddle_surface_offset=float(paddle_surface_offset)
        mount=Rotation.from_euler('z',assembly_yaw_deg,degrees=True).as_matrix()
        mount[np.abs(mount)<1e-14]=0
        self.assembly_rotation=mount
        if assembly_yaw_deg!=0:
            origin=root.find("./joint[@name='panda_joint9']/origin")
            origin.set('xyz',' '.join(map(str,mount@PADDLE_OFFSET)))
            origin.set('rpy',f'{np.pi} 0 {np.deg2rad(assembly_yaw_deg)}')
        if assembly is not None:
            # Whole rigid assembly, expressed at the flange, replaces BOTH
            # old fixture and paddle inertias. Rotate the entire assembly once
            # from its CAD reference frame into the actual mounting frame.
            mass=float(assembly['mass_kg']);com=np.asarray(assembly['com_m'],dtype=float)
            tensor=np.asarray(assembly['inertia_com_kg_m2'],dtype=float)
            if not np.isfinite(mass) or mass<=0 or com.shape!=(3,) or tensor.shape!=(3,3) or not np.isfinite(com).all() or not np.isfinite(tensor).all():
                raise ValueError('Invalid assembly mass properties')
            principal=np.linalg.eigvalsh(tensor)
            if not np.allclose(tensor,tensor.T,atol=1e-12,rtol=0) or principal[0]<=0 or principal[-1]>sum(principal[:2])+1e-12:
                raise ValueError('Assembly COM inertia must be symmetric and physically realizable')
            com=mount@com;tensor=mount@tensor@mount.T
            self.assembly_com_flange=com;self.assembly_inertia_flange=tensor
            for name,m,c,I in [('panda_link8',mass,com,tensor),('paddle',0.,np.zeros(3),np.zeros((3,3)))]:
                ine=links[name].find('inertial')
                ine.find('mass').set('value',repr(m))
                ine.find('origin').set('xyz',' '.join(map(str,c)))
                for i,a in enumerate('xyz'):
                    for j,b in enumerate('xyz'):
                        if i<=j:ine.find('inertia').set('i'+a+b,repr(float(I[i,j])))
            self.pin=pin.buildModelFromXML(ET.tostring(root,encoding='unicode'))
        else:self.pin=pin.buildModelFromUrdf(str(path))
        self.urdf_xml=ET.tostring(root,encoding='unicode')
        self.data=self.pin.createData()
        self.frame=self.pin.getFrameId('paddle')
        self.q_min=self.pin.lowerPositionLimit+.15
        self.q_max=self.pin.upperPositionLimit-.15
        q=ca.SX.sym('q',7);v=ca.SX.sym('v',7);a=ca.SX.sym('a',7)
        R=ca.DM.eye(3);p=ca.DM.zeros(3);w=ca.DM.zeros(3);alpha=w;acc=ca.DM([0,0,9.81])
        forces=[];moments=[];positions=[];axes=[]
        for k,joint in enumerate(root.findall('joint')):
            origin=joint.find('origin')
            offset=R@ca.DM(np.fromstring(origin.get('xyz'),sep=' '))
            rpy=np.fromstring(origin.get('rpy'),sep=' ')
            R=R@ca.DM(Rotation.from_euler('xyz',rpy).as_matrix())
            p=p+offset
            acc=acc+ca.cross(alpha,offset)+ca.cross(w,ca.cross(w,offset))
            if k<7:
                R=R@rz(q[k]);axis=R[:,2]
                alpha=alpha+axis*a[k]+ca.cross(w,axis*v[k]);w=w+axis*v[k]
                axes.append(axis)
            ine=links[joint.find('child').get('link')].find('inertial')
            mass=float(ine.find('mass').get('value'));com=np.fromstring(ine.find('origin').get('xyz'),sep=' ')
            ii=ine.find('inertia').attrib
            I=np.array([[float(ii['i'+i+j]) if 'i'+i+j in ii else float(ii['i'+j+i]) for j in 'xyz'] for i in 'xyz'])
            r=R@ca.DM(com);Iw=R@ca.DM(I)@R.T
            force=mass*(acc+ca.cross(alpha,r)+ca.cross(w,ca.cross(w,r)))
            moment=Iw@alpha+ca.cross(w,Iw@w)+ca.cross(r,force)
            positions.append(p);forces.append(force);moments.append(moment)
        tau=[]
        for k in reversed(range(len(forces))):
            if k<len(forces)-1:
                moments[k]+=moments[k+1]+ca.cross(positions[k+1]-positions[k],forces[k+1])
                forces[k]+=forces[k+1]
            if k<7:tau.insert(0,ca.dot(axes[k],moments[k]))
        tau=ca.vertcat(*tau)
        self.rnea=ca.Function('panda_rnea',[q,v,a],[tau])
        self.mass=ca.Function('panda_mass',[q],[ca.jacobian(tau,a)])
        self.bias=ca.Function('panda_bias',[q,v],[ca.substitute(tau,a,ca.DM.zeros(7))])
        # The normal fit pivots about the measured face center; preserve
        # rigid mounting, marker extrinsics, payload inertia and position.
        surface=p+self.paddle_surface_offset*R[:,2]
        R=R@ca.DM(self.paddle_normal_rotation)
        self.fk=ca.Function('panda_fk',[q],[surface,R[:,2],ca.jacobian(surface,q),ca.jacobian(R[:,2],q),R])
        self.contact_fk=ca.Function('panda_contact_fk',[q],[surface,R[:,2],ca.jacobian(surface,q),ca.jacobian(w,v),R])
        self.origin=np.asarray(self.fk(HOME)[0]).ravel()
        self.geometry=ca.Function('panda_geometry',[q],[ca.horzcat(*positions)])

    def numeric_fk(self,q,v=None):
        pin.forwardKinematics(self.pin,self.data,q)
        pin.updateFramePlacements(self.pin,self.data)
        transform=self.data.oMf[self.frame]
        nominal_normal=transform.rotation[:,2].copy()
        normal=(transform.rotation@self.paddle_normal_rotation)[:,2]
        p=transform.translation.copy()+self.paddle_surface_offset*nominal_normal
        J=pin.computeFrameJacobian(self.pin,self.data,q,self.frame,pin.LOCAL_WORLD_ALIGNED)
        jn=-pin.skew(normal)@J[3:]
        surface_jacobian=J[:3].copy()-self.paddle_surface_offset*pin.skew(nominal_normal)@J[3:]
        return p,normal,surface_jacobian,jn.copy()
