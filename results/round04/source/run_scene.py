"""Run arbitrary prepared meshes: python -m simuscene.run_scene input.json output.json."""
import argparse,json
from dataclasses import asdict
from pathlib import Path
from .core import Object,correct,settle

def main():
    p=argparse.ArgumentParser();p.add_argument('input');p.add_argument('output');a=p.parse_args()
    data=json.loads(Path(a.input).read_text());objects=[Object(**x) for x in data['objects']]
    prepared,events=correct(objects);final,contacts=settle(prepared)
    Path(a.output).write_text(json.dumps({'objects':[asdict(x) for x in final],'events':events,'contacts':contacts,'scope':'geometry-only; no image reconstruction or learned resampling'},indent=2))
if __name__=='__main__':main()
