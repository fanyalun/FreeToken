import json
from pathlib import Path
import sys

from prepare_data import ROOT,grouped_split
from tablemoe_data import write_rows
from freetoken.moe.tablemoe_artifacts import write_json,file_hash

sys.path.insert(0,'/home/fanya/vision2web')
from vision2web.inference.prompts import WEBPAGE_PROMPT

data=Path('/data/fanya/datasets/vision2web_level1/webpage')
rows=[]
for task in sorted(p for p in data.iterdir() if p.is_dir() and (p/'workflow.json').exists()):
    images=[task/'prototypes'/f'{view}.jpg' for view in ['desktop','tablet','mobile']]
    assert all(p.is_file() for p in images)
    rows.append(dict(id='Vision2Web_Level1:'+task.name,index=task.name,dataset='Vision2Web_Level1',
        category='unstratified_user_authorized',metadata=dict(task=task.name,question=WEBPAGE_PROMPT),
        image_ids=[str(p.resolve()) for p in images],
        vlmeval_message=[dict(type='text',value=WEBPAGE_PROMPT)]+[dict(type='image',value=str(p)) for p in images],
        prompt_source='official WEBPAGE_PROMPT; prototype attachment for offline calibration'))
assert len(rows)==100
calibration,evaluation,extra=grouped_split(rows)
assert len(calibration)==60 and len(evaluation)==40
root=ROOT/'prepare/Vision2Web_Level1'
for label,values in [('all',rows),('calibration',calibration),('evaluation',evaluation)]:
    write_rows(root/(label+'.jsonl'),values)
write_json(root/'split.json',dict(seed=42,calibration_fraction=.6,test_fraction=.4,
    stratified=False,reason='official per-task category mapping unavailable; user explicitly authorized task-level split',
    calibration_ids=[r['id'] for r in calibration],evaluation_ids=[r['id'] for r in evaluation],
    calibration_samples=60,evaluation_samples=40,
    image_sharing={key:[r['id']] for r in rows for key in r['image_ids']},**extra))
write_json(root/'protocol.json',dict(official_revision='d80a4a3d0ea7326541ad87314294269a633789e4',
    calibration='official task prompt plus all three prototype images; fixed 128 tokens for routed-input collection',
    evaluation='official ClaudeCodeAdapter agent workflow with tools; not single completion',
    prompt_source_sha256=file_hash('/home/fanya/vision2web/vision2web/inference/prompts.py')))
write_json(root/'complete.json',dict(complete=True,calibration=60,evaluation=40))
print('Vision2Web_Level1 60 40')
