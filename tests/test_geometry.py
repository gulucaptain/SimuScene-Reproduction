import numpy as np
from simuscene.core import Object,stretch,box_iou,probe

def test_stretch_preserves_top_and_moves_bottom():
    o=Object('box',[.2,.3,.4],[0,0,.24]); q,rho=stretch(o,[0,0,-.04])
    assert np.isclose(q.bounds()[1][2],o.bounds()[1][2])
    assert np.isclose(q.bounds()[0][2],0)
    assert np.isclose(rho,.1)

def test_contact_probe_measures_gap():
    o=Object('box',[.2,.3,.4],[0,0,.24]); delta,hit=probe([],o)
    assert hit and abs(delta[2]+.04)<.004

def test_iou_known_volume():
    a=Object('a',[1,1,1],[0,0,0]); b=Object('b',[1,1,1],[.5,0,0])
    assert np.isclose(box_iou(a,b),1/3)

def test_penetration_direction():
    from simuscene.core import resolve_penetration
    o=Object('box',[.2,.2,.2],[0,0,.08]); q,status=resolve_penetration([],o)
    assert status['ok'] and q.bounds()[0][2]>=0

def test_hanging_constraints():
    from simuscene.core import settle
    o=Object('box',[.2,.2,.2],[0,0,.6],anchor=[[-.1,0,.1],[.1,0,.1]])
    final,_=settle([o]); assert np.linalg.norm(np.array(final[0].position)-o.position)<.002

def test_mesh_contact_and_large_error_is_not_faked():
    from simuscene.core import settle,scene,correct
    o=Object('cone',[.2,.2,.3],[0,0,.18],mesh='procedural:cone')
    final,_=settle([o]);assert abs(final[0].position[2]-.15)<.005
    p,_=scene(4,kind='severe');q,events=correct(p)
    assert all(e['branch']=='unresolved_resampling' for e in events)
    assert all(a.size==b.size for a,b in zip(p,q))

def test_depth_alignment_recovers_known_similarity():
    from simuscene.alignment import align_visible
    rng=np.random.default_rng(30); v=rng.uniform(-.5,.5,(80,3));t=np.array([.04,-.02,.03]);points=1.12*v+t
    fit=align_visible(points,v)[0]
    assert abs(fit['scale']-1.12)<1e-4
    assert np.linalg.norm(np.array(fit['translation'])-t)<1e-4

def test_concave_mesh_decomposition(tmp_path):
    import trimesh
    from simuscene.assets import collision_parts
    from simuscene.core import model_for,contact_depths
    # U shape: three disjoint boxes retain an opening that a single convex hull would fill.
    parts=[]
    for ext,pos in [([.06,.3,.3],[-.17,0,0]),([.06,.3,.3],[.17,0,0]),([.4,.3,.06],[0,0,-.12])]:
        mesh=trimesh.creation.box(ext);mesh.apply_translation(pos);parts.append(mesh)
    path=tmp_path/'u.obj';trimesh.util.concatenate(parts).export(path)
    hulls=collision_parts(str(path));assert len(hulls)>1
    assert np.isclose(sum(x[2] for x in hulls),1)

    o=Object('concave',[.4,.3,.3],[0,0,.5],mesh=str(path))
    m,d=model_for([o]);assert m.ngeom>2
    assert all(np.isfinite(fraction) and fraction>0 for _,_,fraction in hulls)
