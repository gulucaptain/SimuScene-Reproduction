"""Depth-visible mesh alignment, paper A.2; camera visibility supplied by caller."""
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import minimize

def align_visible(points,vertices,initial_scale=1.,initial_translation=(0,0,0),prior=.001):
    points=np.asarray(points,dtype=float);vertices=np.asarray(vertices,dtype=float)
    if points.ndim!=2 or vertices.ndim!=2 or points.shape[1]!=3 or vertices.shape[1]!=3:raise ValueError('Expected Nx3 points')
    tree=cKDTree(vertices)
    def loss(x):
        scale=np.exp(x[0]);q=(points-x[1:])/scale
        distance=tree.query(q)[0]*scale
        return distance.mean()+prior*(x[0]-np.log(initial_scale))**2
    candidates=[]
    for multiplier in [.5,.75,1.,1.25]:
        x=np.r_[np.log(initial_scale*multiplier),initial_translation]
        result=minimize(loss,x,method='Powell',options={'maxiter':500,'xtol':1e-7,'ftol':1e-8})
        candidates.append({'scale':float(np.exp(result.x[0])),'translation':result.x[1:].tolist(),'objective':float(result.fun),'success':bool(result.success)})
    # Silhouette ranking requires actual masks/cameras; return all candidates, never pretend it was performed.
    return sorted(candidates,key=lambda x:x['objective'])
