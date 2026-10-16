"""Render HOME and recorded robot poses with the supplied flange-frame CAD STL.

Display only: no physics stepping, robot connection or collision qualification.
"""
import argparse,hashlib,json,os
from pathlib import Path
import xml.etree.ElementTree as ET
os.environ.setdefault('MUJOCO_GL','egl')
import mujoco
import numpy as np
import trimesh
import pinocchio as pin
from scipy.spatial.transform import Rotation
from PIL import Image,ImageDraw,ImageFont
from matplotlib.font_manager import findfont
from arm.model import Panda,HOME


def main():
    p=argparse.ArgumentParser();p.add_argument('--cad',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--stopped-state',type=Path,required=True)
    p.add_argument('--mount-yaw-deg',type=float,default=0.,help='CAD-to-flange rotation about flange +Z')
    args=p.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    cache=Path('build/hardware_pose_meshes').resolve();cache.mkdir(parents=True,exist_ok=True)
    cad=args.cad.resolve();hashes={}
    for line in (cad/'SHA256SUMS').read_text().splitlines():
        expected,name=line.split(maxsplit=1);name=name.lstrip('*')
        actual=hashlib.sha256((cad/name).read_bytes()).hexdigest();assert actual==expected,name;hashes[name]=actual
    mesh=trimesh.load(cad/'payload_assembly_m.stl',force='mesh')
    # Keep every original triangle. Connected parts are grouped only for color;
    # no hull is used as a physical collider and no geometry is simplified.
    groups={k:[] for k in ['adapter','paddle','markers','hardware']}
    for part in mesh.split(only_watertight=False,repair=False):
        e=part.extents
        kind='paddle' if e[0]>.18 else 'adapter' if e[0]>.08 else 'markers' if np.all((e>.0088)&(e<.0092)) else 'hardware'
        groups[kind].append(part)
    colors={'adapter':'.20 .52 .62 1','paddle':'.70 .13 .12 1','markers':'.96 .93 .76 1','hardware':'.65 .69 .73 1'}
    source=Path('runs/20260925T102022.721751Z_native_mesh24_prune_wind/authored_scene.xml')
    root=ET.parse(source).getroot();asset=root.find('asset');world=root.find('worldbody')
    flange=next(b for b in root.iter('body') if b.get('name')=='panda_link8')
    mount=Rotation.from_euler('z',args.mount_yaw_deg,degrees=True)
    attachment=ET.SubElement(flange,'body',name='cad_mount',quat=' '.join(map(str,mount.as_quat(scalar_first=True))))
    for body in root.iter('body'):
        for g in list(body.findall('geom')):
            if g.get('name','').startswith(('fixture_','handle_visual','racket','ball_shape')):body.remove(g)
    for g in root.iter('geom'):g.set('contype','0');g.set('conaffinity','0')
    if root.find('contact') is not None:root.remove(root.find('contact'))
    for name,parts in groups.items():
        path=cache/(name+'.obj');trimesh.util.concatenate(parts).export(path,include_color=False,include_texture=False)
        ET.SubElement(asset,'mesh',name='cad_'+name,file=str(path))
        # Mesh vertices already include the paddle-center offset. Apply only
        # the measured mounting rotation to the complete assembly, once.
        ET.SubElement(attachment,'geom',name='cad_'+name,type='mesh',mesh='cad_'+name,
                      rgba=colors[name],group='2',contype='0',conaffinity='0',mass='0')
    scene=cache/'render_scene.xml';ET.ElementTree(root).write(scene,encoding='unicode')
    model=mujoco.MjModel.from_xml_path(str(scene));data=mujoco.MjData(model)
    model.vis.global_.offwidth=1280;model.vis.global_.offheight=960
    model.vis.headlight.ambient[:]=[.55]*3;model.vis.headlight.diffuse[:]=[.75]*3
    model.vis.headlight.specular[:]=[.2]*3
    option=mujoco.MjvOption();option.geomgroup[3]=0
    robot=Panda();font_path=findfont('DejaVu Sans')
    font=ImageFont.truetype(font_path,28);small=ImageFont.truetype(font_path,22)
    state=json.loads(args.stopped_state.read_text());poses={'home':HOME,'stopped':np.array(state['q'])}
    results={}
    def arrow(renderer,start,end,color,label):
        scn=renderer.scene;g=scn.geoms[scn.ngeom];scn.ngeom+=1
        mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_ARROW,np.zeros(3),np.zeros(3),np.eye(3).ravel(),np.array(color,dtype=np.float32))
        mujoco.mjv_connector(g,mujoco.mjtGeom.mjGEOM_ARROW,.003,start,end);g.label=label
    with mujoco.Renderer(model,960,1280) as renderer:
        for pose,q in poses.items():
            data.qpos[:7]=q;data.qvel[:]=0;mujoco.mj_forward(model,data)
            pin.forwardKinematics(robot.pin,robot.data,q);pin.updateFramePlacements(robot.pin,robot.data)
            T=robot.data.oMf[robot.pin.getFrameId('panda_link8')].homogeneous.copy()
            face=T[:3,3]+T[:3,:3]@mount.apply([.160,0,.0355]);normal=T[:3,:3]@mount.apply([0,0,-1])
            results[pose]=dict(q_deg=np.rad2deg(q).tolist(),flange_in_base=T.tolist(),cad_face_center_base_m=face.tolist(),cad_face_normal_base=normal.tolist())
            views=[('overview',dict(lookat=[.23,-.025,.39],distance=1.65,azimuth=130,elevation=-24)),
                   ('attachment',dict(lookat=(T[:3,3]+face)/2,distance=.58,azimuth=130,elevation=-27)),
                   ('top',dict(lookat=[.27,-.06,.40],distance=1.15,azimuth=90,elevation=-89.9))]
            panels=[]
            for name,settings in views:
                camera=mujoco.MjvCamera()
                for k,v in settings.items():
                    if k=='lookat':camera.lookat[:]=v
                    else:setattr(camera,k,v)
                renderer.update_scene(data,camera=camera,scene_option=option)
                arrow(renderer,face,face+.105*normal,[0,.85,.9,1],'')
                if name=='top':
                    base=np.array([.05,-.30,.03])
                    arrow(renderer,base,base+np.array([.13,0,0]),[.95,.15,.12,1],'X')
                    arrow(renderer,base,base+np.array([0,.13,0]),[.1,.85,.15,1],'Y')
                frame=Image.fromarray(renderer.render());draw=ImageDraw.Draw(frame)
                draw.rectangle((0,0,1280,54),fill=(19,27,36))
                title=('Desired HOME' if pose=='home' else 'Recorded stopped pose')+' | '+name
                draw.text((20,12),title,font=font,fill='white')
                frame.save(out/f'{pose}_{name}.png');panels.append(frame)
            sheet=Image.new('RGB',(1920,1110),(19,27,36));draw=ImageDraw.Draw(sheet)
            draw.text((24,18),'Desired HOME with the actual CAD assembly' if pose=='home' else 'Recorded robot pose after the stopped MPC trial',font=font,fill='white')
            sheet.paste(panels[0],(0,64));sheet.paste(panels[1].resize((640,480)),(1280,64));sheet.paste(panels[2].resize((640,480)),(1280,544))
            draw.text((24,1036),f'Cyan arrow: hitting-face normal. Complete CAD assembly rotated {args.mount_yaw_deg:g} degrees about flange +Z.',font=small,fill=(210,225,237))
            angles=', '.join(f'{v:.1f}' for v in np.rad2deg(q))
            draw.text((24,1070),f'Joint angles [deg] = [{angles}]. Colors indicate components; geometry comes from the STL.',font=small,fill=(210,225,237))
            sheet.save(out/f'{pose}_views.png')
    results.update(cad_source=str(cad),cad_sha256=hashes,mesh_triangles=len(mesh.faces),
        geometry_mode='Display only; exact STL triangles, no collision qualification or physics stepping',
        mesh_attachment='CAD-to-flange mounting rotation applied once at panda_link8',
        mount_yaw_deg=args.mount_yaw_deg,mount_rotation_matrix=mount.as_matrix().tolist(),
        cad_face_flange_m=mount.apply([.160,0,.0355]).tolist(),
        previously_compiled_mpc_face_flange_m=[.160,0,.034],
        blade_half_thickness_m=.0045,physical_motion_during_render=False)
    (out/'geometry.json').write_text(json.dumps(results,indent=2)+'\n');print(out/'home_views.png')


if __name__=='__main__':main()
