import json,hashlib,platform
from pathlib import Path
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from .core import Object
from .assets import normalized_mesh
import trimesh

ROOT=Path(__file__).resolve().parents[2]
def draw(ax,objects,title):
    for i,obj in enumerate(objects):
        o=Object(**obj)
        if o.mesh:v,f=normalized_mesh(o.mesh);v=v*np.array(o.size)
        else:
            mesh=trimesh.creation.box(o.size);v,f=mesh.vertices,mesh.faces
        v=v@o.rotation().T+o.position
        ax.add_collection3d(Poly3DCollection(v[f],alpha=.8,facecolor=plt.cm.tab10(i%10),edgecolor='k',linewidth=.1))
    ax.set(xlim=(-.85,.85),ylim=(-.45,.65),zlim=(0,.8),title=title,xlabel='x (m)',ylabel='y (m)');ax.set_box_aspect([1.7,1.1,.8]);ax.view_init(22,-65)

def main():
    directory=ROOT/'results/round05';rows=json.loads((directory/'raw.json').read_text());summary=json.loads((directory/'summary.json').read_text())
    kinds=['short','penetrating','stack','rotated','pose_error','mesh','severe','hanging']; methods=['settle','correct:0.15']
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    for j,metric in enumerate(['aabb_iou','size_relative_error']):
        for k,m in enumerate(methods):
            means=[];errors=[]
            for kind in kinds:
                x=np.array([r[metric] for r in rows if r['kind']==kind and r['method']==m]);means.append(x.mean());errors.append(1.96*x.std(ddof=1)/np.sqrt(len(x)))
            axes[j].bar(np.arange(len(kinds))+(k-.5)*.35,means,.35,yerr=errors,label=m)
        axes[j].set_xticks(np.arange(len(kinds)),kinds,rotation=35,ha='right');axes[j].set_title(metric+'; 95% normal CI across seeds');axes[j].legend()
    fig.savefig(directory/'comparison.png',dpi=170);plt.close(fig)
    fig=plt.figure(figsize=(13,8),layout='constrained')
    for j,kind in enumerate(['stack','mesh']):
        base=next(r for r in rows if r['seed']==0 and r['kind']==kind and r['method']=='settle'); corrected=next(r for r in rows if r['seed']==0 and r['kind']==kind and r['method']=='correct:0.15')
        for k,(objects,title) in enumerate([(base['initial'],'Initial'),(base['final'],'Settle only'),(corrected['final'],'Diagnostic + stretch')]):draw(fig.add_subplot(2,3,j*3+k+1,projection='3d'),objects,kind+': '+title)
    fig.savefig(directory/'scenes.png',dpi=170);plt.close(fig)
    events=Counter(e['branch'] for r in rows for e in r['events']);(directory/'event_counts.json').write_text(json.dumps(events,indent=2))
    lines=['# Local results — not the paper benchmark','',f'{len(rows)} scene-method runs; 64 seeds, 8 objects per scene; 8 CPU workers.','', '| Scene | Settle IoU | Correct IoU | Difference | Unresolved fraction |','|---|---:|---:|---:|---:|']
    for kind in kinds:
        a=summary['settle/'+kind];b=summary['correct:0.15/'+kind];lines.append(f"| {kind} | {a['aabb_iou']:.5f} | {b['aabb_iou']:.5f} | {b['aabb_iou']-a['aabb_iou']:+.5f} | {b['unresolved_fraction']:.3f} |")
    lines+=['','Pose-only errors are an explicit counterexample: shape correction can damage correct geometry. Severe errors remain unresolved without a learned resampling model. Convex test meshes use direct hulls. The final implementation supports V-HACD for concave file inputs; this additional path is checked separately, not included in the synthetic benchmark.','', 'Figures: [comparison](comparison.png), [scenes](scenes.png). Raw per-object states: [raw.json](raw.json).']
    (directory/'REPORT.md').write_text('\n'.join(lines));print('\n'.join(lines))
if __name__=='__main__':main()
