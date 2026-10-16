"""GPU off-screen rendering of recorded Newton coordinates; no physics replay."""
import argparse,json,os
from pathlib import Path
import xml.etree.ElementTree as ET
os.environ.setdefault('MUJOCO_GL','egl')
import mujoco
import numpy as np
import imageio.v2 as imageio
from PIL import Image,ImageDraw,ImageFont


def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--still',action='store_true');args=p.parse_args();run=args.run
    frames=np.load(run/'frames.npz');cfg=json.loads((run/'config.json').read_text());metrics=json.loads((run/'metrics.json').read_text())
    m=mujoco.MjModel.from_xml_path(str(run/'engine_scene.xml'));d=mujoco.MjData(m)
    appearance={g.get('name'):g.get('rgba') for g in ET.parse(run/'authored_scene.xml').getroot().iter('geom') if g.get('name')}
    for i in range(m.ngeom):
        name=m.geom(i).name.rsplit('/',1)[-1];base=name.rsplit('_',1)[0]
        if appearance.get(base):m.geom_rgba[i]=np.fromstring(appearance[base],sep=' ')
        if base.startswith('collision_link'):m.geom_group[i]=3
    m.vis.headlight.ambient[:]=[.55]*3;m.vis.headlight.diffuse[:]=[.75]*3;m.vis.headlight.specular[:]=[.15]*3
    m.vis.global_.offwidth=1280;m.vis.global_.offheight=960
    cam=mujoco.MjvCamera();cam.lookat[:]=[.28,-.06,.74];cam.distance=2.15;cam.azimuth=130;cam.elevation=-18
    opt=mujoco.MjvOption();opt.geomgroup[3]=0
    from matplotlib.font_manager import findfont
    font_path=findfont('DejaVu Sans')
    font=ImageFont.truetype(font_path,23)
    small=ImageFont.truetype(font_path,18)
    selected=np.unique(np.linspace(0,len(frames['t'])-1,6,dtype=int));stills=[]
    writer=None if args.still else imageio.get_writer(run/'video.mp4',fps=int(frames['fps']),codec='libx264',quality=8,pixelformat='yuv420p')
    with mujoco.Renderer(m,960,1280) as renderer:
        for i,q in enumerate(frames['qpos']):
            if args.still and i not in selected:continue
            d.qpos[:]=q;d.qvel[:]=0;mujoco.mj_forward(m,d)
            renderer.update_scene(d,camera=cam,scene_option=opt);frame=Image.fromarray(renderer.render());draw=ImageDraw.Draw(frame)
            draw.rectangle((0,0,1280,94),fill=(20,27,35))
            draw.text((24,13),'Newton · Panda ball juggling · 1 SQP / 50 OSQP',font=font,fill='white')
            text=f"t = {frames['t'][i]:5.2f} s    |    plant 0.5 ms    |    MPC {1000*cfg['mpc']['control_dt']:g} ms    |    wind: {cfg['arguments']['wind']}"
            draw.text((24,52),text,font=small,fill=(200,218,230))
            if writer:writer.append_data(np.asarray(frame))
            if i in selected:stills.append(frame.copy())
            if i==0:frame.save(run/'preview.png')
    if writer:writer.close()
    sheet=Image.new('RGB',(1280*3//2,960),color=(20,27,35))
    for i,frame in enumerate(stills):sheet.paste(frame.resize((640,480)),((i%3)*640,(i//3)*480))
    sheet.save(run/'contact_sheet.png')
    (run/'rendering.json').write_text(json.dumps(dict(backend='MuJoCo EGL off-screen renderer of Newton-generated scene',
        frames=len(frames['t']),fps=int(frames['fps']),resolution=[1280,960],
        timestamps_source='frames.npz',physics_rerun=False,appearance_only_changes=True),indent=2)+'\n')
    print(run/'video.mp4' if writer else run/'contact_sheet.png')

if __name__=='__main__':main()
