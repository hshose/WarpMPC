"""Author measured elliptical racket geometry without altering controller geometry."""
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation


def author_plant_geometry(path, robot, plant):
    root = ET.parse(path)
    geom = root.find(".//geom[@name='racket']")
    nominal = robot.paddle_normal_rotation
    rotation = nominal@Rotation.from_rotvec(np.deg2rad(plant['normal_delta_rotvec_deg'])).as_matrix()
    center = (np.array([0.,0.,robot.paddle_surface_offset])
              + nominal@np.asarray(plant['center_offset_local_m'])
              - robot.paddle_surface_offset*rotation[:,2])
    geom.set('pos',' '.join(map(str,center)))
    geom.set('quat',' '.join(map(str,Rotation.from_matrix(rotation).as_quat(scalar_first=True))))
    axes = plant['paddle_axes_m']
    # Convex polygonal cylinder; 128 sides bounds radial discretization at 24 um.
    n=128;angle=np.arange(n)*2*np.pi/n
    vertices=np.array([[axes[0]*np.cos(t),axes[1]*np.sin(t),z]
                       for z in [-robot.paddle_surface_offset,robot.paddle_surface_offset] for t in angle])
    ET.SubElement(root.find('asset'),'mesh',name='empirical_racket',vertex=' '.join(map(str,vertices.ravel())))
    geom.set('type','mesh');geom.set('mesh','empirical_racket');geom.attrib.pop('size',None)
    root.write(path,encoding='unicode')
    return rotation,center
