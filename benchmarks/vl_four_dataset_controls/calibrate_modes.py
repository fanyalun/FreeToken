from collections import defaultdict
from datetime import datetime, timezone
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import shutil
import sys
import time
import types

from run_cell import ROOT,SNAPSHOT,MODEL,processor
from tablemoe_data import read_rows
from freetoken.moe.tablemoe_artifacts import write_json


def select(name):
    rows=read_rows(ROOT/'prepare'/name/'calibration.jsonl')
    evaluation=read_rows(ROOT/'prepare'/name/'evaluation.jsonl')
    if {row['id'] for row in rows} & {row['id'] for row in evaluation}:
        raise ValueError('MoDES calibration/evaluation identities overlap')
    if {image for row in rows for image in row['image_ids']} & {image for row in evaluation for image in row['image_ids']}:
        raise ValueError('MoDES calibration/evaluation image provenance overlaps')
    groups=defaultdict(list)
    for row in rows:
        groups[row['category']].append(row)
    rng=random.Random(42)
    selected=[]
    for category in sorted(groups):
        candidates=sorted(groups[category],key=lambda r:r['id'])
        rng.shuffle(candidates)
        selected.extend(candidates[:2])
    reference_kind='ground_truth_from_frozen_calibration_partition'
    if name=='Vision2Web_Level1':
        from run_cell import asset_root
        collection=asset_root(name)/'collect'
        marker=collection/'complete.json'
        if not marker.exists() or not json.loads(marker.read_text()).get('complete'):
            raise ValueError('Vision2Web teacher references require completed full-route collection')
        predictions=read_rows(collection/'requests.jsonl')
        if [row['id'] for row in predictions] != [row['id'] for row in rows]:
            raise ValueError('Vision2Web teacher predictions differ from frozen calibration membership/order')
        teacher={row['id']:row['prediction_text'] for row in predictions}
        selected=copy.deepcopy(selected)
        for row in selected:
            answer=teacher[row['id']]
            if not answer.strip():
                raise ValueError('empty Vision2Web teacher generation')
            row['metadata']['answer']=answer
        reference_kind='full_route_fixed_128_token_collection_predictions_from_calibration_only'
    write_json(ROOT/'calibration'/name/'selection.json',dict(seed=42,
        selected_ids=[r['id'] for r in selected],category_counts={k:min(2,len(v)) for k,v in groups.items()},
        reference_kind=reference_kind,heldout_samples_used=0))
    return selected


def load_calibrator(name):
    base=ROOT.parents[0]/'tablemoe_sensitivity_preserve_vlmeval_20261110'
    for source,dest in [(base/'checkpoint_binding.json',ROOT/'checkpoint_binding.json'),
                        (base/'calibration_support/modes_search.py',ROOT/'calibration_support/modes_search.py')]:
        dest.parent.mkdir(exist_ok=True)
        if not dest.exists():
            shutil.copyfile(source,dest)
    source=ROOT/'sources/calibrate_source.py'
    if not source.exists():
        shutil.copyfile(base/'calibrate.py',source)
    source_hash=hashlib.sha256(source.read_bytes()).hexdigest()
    write_json(ROOT/'calibration'/name/'source.json',dict(source=str(source),sha256=source_hash,
        original_source=str(base/'calibrate.py'),wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    shim=types.ModuleType('run_experiment')
    shim.ALIASES={name:name}
    shim.MODEL=MODEL
    shim.calibration_rows=select
    shim.now=lambda:datetime.now(timezone.utc).isoformat()
    shim.save=write_json
    previous=sys.modules.get('run_experiment')
    original_path=sys.path.copy()
    sys.modules['run_experiment']=shim
    spec=importlib.util.spec_from_file_location('frozen_modes_calibration',source)
    module=importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path[:]=original_path
        if previous is None:
            sys.modules.pop('run_experiment',None)
        else:
            sys.modules['run_experiment']=previous
    module.ROOT,module.SNAPSHOT,module.MODEL=ROOT,SNAPSHOT,MODEL
    module.ALIASES={name:name}
    module.calibration_rows=select
    return module


def main(name):
    module=load_calibrator(name)
    import modes_calibration
    original=modes_calibration.make_inputs
    modes_calibration.make_inputs=lambda proc,row,kwargs,limit:original(proc,row,processor(name),65536)
    module.main(name,.6,time.time()+4*3600,scopes=('prefill_decode',))


if __name__=='__main__':
    main(sys.argv[1])
