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
