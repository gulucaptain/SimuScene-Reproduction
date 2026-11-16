from dataclasses import dataclass, field, asdict
from copy import deepcopy
import json
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation

@dataclass
class Object:
    name: str
    size: list  # full box dimensions in metres
    position: list
    quat: list = field(default_factory=lambda: [1., 0., 0., 0.])  # MuJoCo wxyz
    mass: float = 1.
    anchor: list | None = None
    enabled: bool = True

    def rotation(self):
        return Rotation.from_quat(np.roll(self.quat, -1)).as_matrix()

    def bounds(self):
        half = np.abs(self.rotation()) @ (np.asarray(self.size)/2)
        return np.asarray(self.position)-half, np.asarray(self.position)+half


def model_for(objects, active=None, timestep=1/480):
    """Static floor and CPU rigid dynamics; SI units throughout."""
    fmt = lambda v: ' '.join(f'{x:.12g}' for x in v)
    bodies=[]
    for i,o in enumerate(objects):
        dynamic = o.enabled and (active is None or i in active)
        joint='<freejoint/>' if dynamic else ''
        bodies.append(f'<body name="{o.name}" pos="{fmt(o.position)}" quat="{fmt(o.quat)}">{joint}<geom name="g{i}" type="box" size="{fmt(np.asarray(o.size)/2)}" mass="{o.mass}"/></body>')
    xml=f'''<mujoco><option timestep="{timestep}" gravity="0 0 -9.81" integrator="implicitfast" iterations="100"/><default><geom friction="1 .95 .01" condim="4" solref=".01 2" margin="0"/></default><worldbody><geom name="floor" type="plane" size="5 5 .1"/>{''.join(bodies)}</worldbody></mujoco>'''
    m=mujoco.MjModel.from_xml_string(xml); d=mujoco.MjData(m); mujoco.mj_forward(m,d)
    return m,d


def states(objects,m,d):
    out=deepcopy(objects)
    for o in out:
        bid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,o.name)
        o.position=d.xpos[bid].tolist(); o.quat=d.xquat[bid].tolist()
    return out


def settle(objects,seconds=5.):
    m,d=model_for(objects)
    mujoco.mj_step(m,d,nstep=round(seconds/m.opt.timestep)); mujoco.mj_forward(m,d)
    return states(objects,m,d), contact_depths(m,d)


def contact_depths(m,d):
    depths={}
    for c in d.contact:
        for g in c.geom:
            name=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,int(m.geom_bodyid[g]))
            if name and name!='world': depths[name]=max(depths.get(name,0.),max(0.,-float(c.dist)))
    return depths


def box_iou(a,b):
    al,ah=a.bounds(); bl,bh=b.bounds()
    v=np.prod(np.maximum(0,np.minimum(ah,bh)-np.maximum(al,bl)))
    return float(v/(np.prod(ah-al)+np.prod(bh-bl)-v+1e-15))


def evaluate(initial,final,truth,depths):
    drift=np.array([np.linalg.norm(np.array(a.position)-b.position) for a,b in zip(initial,final)])
    return {'stability_all':float(np.mean(drift<.05)), 'drift_mean_m':float(drift.mean()),
            'aabb_iou':float(np.mean([box_iou(a,b) for a,b in zip(final,truth)])),
            'size_relative_error':float(np.mean([np.linalg.norm(np.array(a.size)-b.size)/np.linalg.norm(b.size) for a,b in zip(final,truth)])),
            'penetrating_fraction':float(np.mean([depths.get(o.name,0)>.002 for o in final])),
            'max_penetration_m':float(max(depths.values(),default=0)), 'objects':len(final)}


def scene(seed,n=8):
    rng=np.random.default_rng(seed); truth=[]; predicted=[]
    for i in range(n):
        size=rng.uniform([.10,.10,.15],[.22,.22,.35]); pos=np.array([(i%4)*.4-.6,(i//4)*.4-.2,size[2]/2])
        t=Object(f'object_{i}',size.tolist(),pos.tolist()); truth.append(t)
        p=deepcopy(t); missing=rng.uniform(.01,.10)*size[2]
        p.size[2]-=missing; p.position[2]+=missing/2  # hidden bottom missing, visible top fixed
        predicted.append(p)
    return predicted,truth
