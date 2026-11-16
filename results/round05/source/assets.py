"""User meshes are centered/aligned once; collision uses their convex hull."""
from functools import lru_cache
from pathlib import Path
import numpy as np
import trimesh

@lru_cache(maxsize=128)
def normalized_mesh(path):
    if path=='procedural:cylinder': m=trimesh.creation.cylinder(radius=.5,height=1,sections=24)
    elif path=='procedural:cone': m=trimesh.creation.cone(radius=.5,height=1,sections=24)
    elif path=='procedural:ellipsoid': m=trimesh.creation.icosphere(subdivisions=2)
    else:
        m=trimesh.load(Path(path).expanduser(),force='mesh')
        if not isinstance(m,trimesh.Trimesh):raise ValueError('Expected a triangle mesh')
        # Canonical oriented bounding box, matching size axes used by core.stretch.
        m.apply_transform(np.linalg.inv(m.bounding_box_oriented.primitive.transform))
    v=np.array(m.vertices); v-=(v.max(0)+v.min(0))/2; ext=np.ptp(v,axis=0)
    if np.any(ext<1e-8):raise ValueError('Mesh must have nonzero extent on all axes')
    return v/ext,np.asarray(m.faces)

COLLISION_AUDIT = {}

@lru_cache(maxsize=128)
def collision_parts(path):
    """V-HACD parts remain in the same canonical coordinate frame."""
    vertices,faces=normalized_mesh(path)
    mesh=trimesh.Trimesh(vertices=vertices,faces=faces,process=True)
    if mesh.is_convex:return ((vertices,faces,1.),)
    parts=mesh.convex_decomposition(maxConvexHulls=16,resolution=100000)
    if isinstance(parts,trimesh.Trimesh):parts=[parts]
    input_count=len(parts); solids=[]
    for part in parts:
        points=np.unique(np.asarray(part.vertices),axis=0)
        if len(points)<4 or np.linalg.matrix_rank(points-points.mean(0),tol=1e-9)<3:continue
        hull=trimesh.convex.convex_hull(points)
        if abs(hull.volume)>1e-10:solids.append(hull)
    if not solids:raise ValueError('V-HACD returned no non-degenerate volume')
    parts=solids
    COLLISION_AUDIT[path]={'returned_parts':input_count,'solid_parts':len(parts),'discarded_degenerate_parts':input_count-len(parts),'maxConvexHulls':16,'resolution':100000}
    volumes=np.array([abs(x.volume) for x in parts]);volumes/=volumes.sum()
    return tuple((np.asarray(p.vertices),np.asarray(p.faces),float(w)) for p,w in zip(parts,volumes))
