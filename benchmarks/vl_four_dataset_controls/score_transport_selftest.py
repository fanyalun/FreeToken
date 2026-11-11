import io
import json
from pathlib import Path

import score_native as scoring


class Opener:
    def __init__(self, text):
        self.text=text
        self.calls=[]

    def open(self, request, timeout):
        self.calls.append(json.loads(request.data))
        return io.BytesIO(json.dumps(dict(choices=[dict(message=dict(content=self.text))])).encode())


root=scoring.ROOT/'validation/judge_transport_mock'
root.mkdir(parents=True,exist_ok=True)
scoring.time.sleep=lambda value:None
checks=[]
for dataset,valid,invalid in [
    ('BabyVision','True','I cannot decide'),
    ('MMLongBench_DOC','Extracted answer: 42\nAnswer format: Integer','42'),
    ('MMMU_DEV_VAL','A',''),
]:
    judge=scoring.StrictJudge(1,dataset,root)
    judge.sample_id='fixture'
    judge.opener=Opener(valid)
    assert judge.generate('native prompt')==valid
    assert len(judge.opener.calls)==1
    assert judge.opener.calls[0]['max_tokens']==judge.max_tokens
    assert judge.opener.calls[0]['messages']==[dict(role='user',content='native prompt')]
    judge.opener=Opener(invalid)
    try:
        judge.generate('native prompt')
    except RuntimeError:
        pass
    else:
        raise AssertionError('invalid judge response accepted')
    assert len(judge.opener.calls)==3
    checks.append(dict(dataset=dataset,valid_response=True,invalid_response_rejected=True,attempts=3))
scoring.save(root/'complete.json',dict(passed=True,checks=checks,gpu_used=False))
print(json.dumps(checks))
