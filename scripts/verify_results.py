"""Verify archived source provenance and numerical summaries without rerunning experiments."""
import hashlib,json,math
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[1]
for folder in sorted((root/'results').glob('round[0-9][0-9]')):
    manifest=json.loads((folder/'manifest.json').read_text())
    for name,digest in manifest['source_sha256'].items():
        assert hashlib.sha256((folder/'source'/name).read_bytes()).hexdigest()==digest,(folder,name)
rows=json.loads((root/'results/round05/raw.json').read_text())
summary=json.loads((root/'results/round05/summary.json').read_text())
assert len(rows)==2048
for group,metrics in summary.items():
    selected=[r for r in rows if r['method']+'/'+r['kind']==group]
    assert len(selected)==64,group
    for key,value in metrics.items():
        values=[r[key] for r in selected if r[key] is not None]
        actual=float(np.mean(values)) if values else None
        assert (value is None and actual is None) or math.isclose(value,actual,rel_tol=1e-10,abs_tol=1e-12),(group,key)
print('Verified five source snapshots and final raw-result summaries.')
