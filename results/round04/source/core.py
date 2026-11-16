from dataclasses import dataclass, field, asdict
from copy import deepcopy
import json
import numpy as np
import mujoco
from .assets import normalized_mesh
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
    mesh: str | None = None

    def rotation(self):
        return Rotation.from_quat(np.roll(self.quat, -1)).as_matrix()

    def bounds(self):
        half = np.abs(self.rotation()) @ (np.asarray(self.size)/2)
        return np.asarray(self.position)-half, np.asarray(self.position)+half


def model_for(objects, active=None, timestep=1/480):
    """Static floor and CPU rigid dynamics; SI units throughout."""
    fmt = lambda v: ' '.join(f'{x:.12g}' for x in v)
    bodies=[]; assets=[]
    for i,o in enumerate(objects):
        dynamic = o.enabled and (active is None or i in active)
        joint='<freejoint/>' if dynamic else ''
        if o.mesh:
            vertices,faces=normalized_mesh(o.mesh)
            assets.append(f'<mesh name="mesh{i}" vertex="{fmt((vertices*np.array(o.size)).ravel())}" face="{fmt(faces.ravel())}"/>')
            geom=f'<geom name="g{i}" type="mesh" mesh="mesh{i}" mass="{o.mass}"/>'
        else:geom=f'<geom name="g{i}" type="box" size="{fmt(np.asarray(o.size)/2)}" mass="{o.mass}"/>'
        bodies.append(f'<body name="{o.name}" pos="{fmt(o.position)}" quat="{fmt(o.quat)}">{joint}{geom}</body>')
    anchors=[]
    for i,o in enumerate(objects):
        if o.anchor and o.enabled and (active is None or i in active):
            for pin in o.anchor: anchors.append(f'<connect body1="{o.name}" anchor="{fmt(pin)}" solref=".01 1"/>')
    equality='<equality>'+''.join(anchors)+'</equality>' if anchors else ''
    xml=f'''<mujoco><option timestep="{timestep}" gravity="0 0 -9.81" integrator="implicitfast" iterations="100"/><default><geom friction="1 .95 .01" condim="4" solref=".01 2" margin="0"/></default><asset>{"".join(assets)}</asset><worldbody><geom name="floor" type="plane" size="5 5 .1"/>{''.join(bodies)}</worldbody>{equality}</mujoco>'''
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


def scene(seed,n=8,kind="short"):
    if kind!="short": return difficult_scene(seed,n,kind)
    rng=np.random.default_rng(seed); truth=[]; predicted=[]
    for i in range(n):
        size=rng.uniform([.10,.10,.15],[.22,.22,.35]); pos=np.array([(i%4)*.4-.6,(i//4)*.4-.2,size[2]/2])
        t=Object(f'object_{i}',size.tolist(),pos.tolist()); truth.append(t)
        p=deepcopy(t); missing=rng.uniform(.01,.10)*size[2]
        p.size[2]-=missing; p.position[2]+=missing/2  # hidden bottom missing, visible top fixed
        predicted.append(p)
    return predicted,truth


def stretch(o, displacement):
    """Eqs. 8, 19–24: preserve anti-gravity OBB face, stretch canonical axis."""
    out=deepcopy(o); g=np.array([0.,0.,-1.]); r=o.rotation()
    axis=int(np.argmax(np.abs(r.T@g))); u=r[:,axis]
    if u@g<0:u=-u
    eta=float(g@np.asarray(displacement)/o.size[axis]); factor=1+eta
    if factor<=0:raise ValueError('Non-positive stretch factor')
    out.size[axis]*=factor
    out.position=(np.array(o.position)+.5*eta*o.size[axis]*u).tolist()
    return out,abs(eta)


def probe(placed,o,max_seconds=3):
    objects=placed+[deepcopy(o)]; active=len(placed); m,d=model_for(objects,{active})
    bid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,o.name)
    start=d.xpos[bid].copy()
    for _ in range(round(max_seconds/m.opt.timestep)):
        mujoco.mj_forward(m,d)
        if any(bid in m.geom_bodyid[c.geom] for c in d.contact):break
        mujoco.mj_step(m,d)
    else:return np.zeros(3),False
    return d.xpos[bid]-start,True


def resolve_penetration(placed,o):
    out=deepcopy(o); travelled=0.
    for iteration in range(200):
        m,d=model_for(placed+[out],{len(placed)})
        bid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,out.name)
        hits=[]
        for c in d.contact:
            if c.dist>=0 or bid not in m.geom_bodyid[c.geom]:continue
            sign=-1 if m.geom_bodyid[c.geom[0]]==bid else 1
            environment=0 in m.geom_bodyid[c.geom]
            hits.append((environment,-float(c.dist),sign*np.array(c.frame[:3])))
        if not hits:return out,{'ok':True,'iterations':iteration,'travelled':travelled}
        top=max(x[0] for x in hits); group=[x for x in hits if x[0]==top]
        normal=sum(depth*normal for _,depth,normal in group); norm=np.linalg.norm(normal)
        if norm<1e-10:return out,{'ok':False,'reason':'opposed_contact_normals','travelled':travelled}
        delta=(max(x[1] for x in group)+.007)*normal/norm
        # Anchored objects must not be translated away from their initial wall pins.
        if out.anchor:return out,{'ok':False,'reason':'anchored_penetration_requires_constrained_solver','travelled':travelled}
        out.position=(np.array(out.position)+delta).tolist(); travelled+=np.linalg.norm(delta)
        if travelled>.3:return out,{'ok':False,'reason':'displacement_bound','travelled':travelled}
    return out,{'ok':False,'reason':'iteration_bound','travelled':travelled}


def correct(objects,threshold=.15,resolve=True,resampler=None):
    placed=[]; events=[]
    for i in sorted(range(len(objects)),key=lambda i:objects[i].bounds()[0][2]):
        original=deepcopy(objects[i]); o=deepcopy(original)
        if o.anchor:
            # Let pin constraints settle orientation; no free-body shape correction.
            placed.append(o); events.append({'name':o.name,'branch':'anchored','rho':0.}); continue
        if resolve:o,status=resolve_penetration(placed,o)
        else:status={'ok':True}
        if not status['ok']:
            placed.append(original); events.append({'name':o.name,'branch':'resolution_failed','rho':None,**status}); continue
        gravity,hit=probe(placed,o)
        delta=np.array(o.position)-original.position+gravity
        candidate,rho=stretch(original,delta)
        branch='stretch' if hit and rho<threshold else 'unresolved_resampling'
        if branch=='stretch':o=candidate
        elif resampler is not None and hit:
            o=resampler(original,candidate)
            if not isinstance(o,Object):raise TypeError('Resampler must return Object')
            branch='external_resampling'
        events.append({'name':o.name,'displacement':delta.tolist(),'rho':rho,'branch':branch,'resolution':status})
        placed.append(o)
    byname={o.name:o for o in placed}
    return [byname[o.name] for o in objects],events


def difficult_scene(seed,n,kind):
    predicted,truth=scene(seed,n)
    rng=np.random.default_rng(seed+100000)
    if kind=='penetrating':
        for p,t in zip(predicted,truth):
            p.size=t.size.copy(); p.size[2]*=1.08; p.position=t.position.copy(); p.position[2]-=.04*t.size[2]
    elif kind=='pose_error':
        for p,t in zip(predicted,truth):p.size=t.size.copy();p.position=t.position.copy();p.position[2]+=.02
    elif kind=='stack':
        for i,(p,t) in enumerate(zip(predicted,truth)):
            if i%2:
                below=truth[i-1];t.position=below.position.copy();t.position[2]+=below.size[2]/2+t.size[2]/2
                p.position=t.position.copy();p.position[2]+=(t.size[2]-p.size[2])/2
    elif kind=='rotated':
        for p,t in zip(predicted,truth):
            q=Rotation.from_euler('z',rng.uniform(-np.pi,np.pi)).as_quat();p.quat=np.roll(q,1).tolist();t.quat=p.quat.copy()
    elif kind=='severe':
        for p,t in zip(predicted,truth):
            p.size=t.size.copy();p.size[2]*=.65;p.position=t.position.copy();p.position[2]+=.175*t.size[2]
    elif kind=='mesh':
        for i,(p,t) in enumerate(zip(predicted,truth)):
            p.mesh=['procedural:cylinder','procedural:cone','procedural:ellipsoid'][i%3];t.mesh=p.mesh
    elif kind=='hanging':
        for p,t in zip(predicted,truth):
            p.size=t.size.copy();p.position=t.position.copy();p.position[2]+=.6;t.position=p.position.copy()
            p.anchor=[[-p.size[0]/2,0,p.size[2]/2],[p.size[0]/2,0,p.size[2]/2]];t.anchor=deepcopy(p.anchor)
    else:raise ValueError(kind)
    return predicted,truth
