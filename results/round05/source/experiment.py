import argparse,json,time,os,platform,hashlib
from pathlib import Path
from dataclasses import asdict
from concurrent.futures import ProcessPoolExecutor
from .core import scene,settle,evaluate,correct
import numpy as np

def job(args):
    seed,method,kind=args; initial,truth=scene(seed,kind=kind)
    start=time.perf_counter()
    threshold=float(method.split(":")[1]) if ":" in method else .15
    prepared,events=correct(initial,threshold=threshold) if method.startswith("correct") else (initial,[])
    final,depths=settle(prepared)
    metrics=evaluate(prepared,final,truth,depths)
    from .core import model_for,contact_depths
    mm,dd=model_for(prepared); before=contact_depths(mm,dd)
    eligible=[i for i,o in enumerate(prepared) if not o.anchor and before.get(o.name,0)<=.002]
    metrics['stability_eligible_count']=len(eligible)
    metrics['paper_stability_eligible']=float(np.mean([np.linalg.norm(np.array(final[i].position)-prepared[i].position)<.05 for i in eligible])) if eligible else None
    metrics['unresolved_fraction']=sum(e['branch'] in ['unresolved_resampling','resolution_failed'] for e in events)/len(initial)
    return dict(seed=seed,method=method,kind=kind,seconds=time.perf_counter()-start,events=events,**metrics,
                initial=[asdict(x) for x in initial],final=[asdict(x) for x in final],truth=[asdict(x) for x in truth])

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--round',type=int,default=1); ap.add_argument('--seeds',type=int,default=8); ap.add_argument('--workers',type=int,default=8); a=ap.parse_args()
    root=Path(__file__).resolve().parents[2]; out=root/'results'/f'round{a.round:02d}'; out.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=a.workers) as pool: rows=list(pool.map(job,[(s,m,k) for s in range(a.seeds) for k in (['short'] if a.round<3 else (['short','penetrating','pose_error','stack','rotated','hanging']+(['mesh','severe'] if a.round>=4 else []))) for m in (['settle'] if a.round==1 else (['settle','correct'] if a.round<5 else ['settle','correct:0.05','correct:0.15','correct:0.30']))]))
    (out/'raw.json').write_text(json.dumps(rows,indent=2)); summary={m:{k:(float(np.mean([r[k] for r in rows if r['method']+'/'+r['kind']==m and r[k] is not None])) if any(r[k] is not None for r in rows if r['method']+'/'+r['kind']==m) else None) for k in ['stability_all','aabb_iou','size_relative_error','penetrating_fraction','seconds','paper_stability_eligible','stability_eligible_count','unresolved_fraction']} for m in sorted({r['method']+'/'+r['kind'] for r in rows})}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)); (out/'manifest.json').write_text(json.dumps({'args':vars(a),'elapsed':time.perf_counter()-start,'platform':platform.platform(),'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}},indent=2))
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
