import argparse
import copy
import json
from pathlib import Path
import time

import score_native as scoring


def main(name):
    actual=scoring.ROOT
    candidates=scoring.rows(actual/'prepare'/name/'evaluation.jsonl')
    if name=='MMMU_DEV_VAL':
        choice=next(row for row in candidates if row['metadata'].get('question_type')=='multiple-choice')
        opened=next(row for row in candidates if row['metadata'].get('question_type')!='multiple-choice')
        expected=[choice,opened]
    elif name=='MMLongBench_DOC':
        expected=[row for row in candidates if str(row['metadata']['answer'])!='Not answerable'][:2]
    else:
        expected=candidates[:2]
    expected=copy.deepcopy(expected)
    root=actual/'validation/native_mock'/name/str(time.time_ns())
    scoring.ROOT=root
    prepare=root/'prepare'/name
    prepare.mkdir(parents=True,exist_ok=True)
    (prepare/'evaluation.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in expected))
    cell=root/'cells'/name/'mock'
    cell.mkdir(parents=True,exist_ok=True)
    predictions=[]
    for row in expected:
        answer=str(row['metadata']['answer'])
        text='The final answer is '+answer
        if name=='BabyVision':
            text='\\boxed{'+answer+'}'
        predictions.append(dict(id=row['id'],prediction_text=text,natural_eos=True,finish_reason='stop'))
    (cell/'requests.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in predictions))
    scoring.save(cell/'generation_complete.json',dict(complete=True))
    answers={row['id']:str(row['metadata']['answer']) for row in expected}
    def generate(self,prompt,**kwargs):
        if name=='BabyVision':
            return 'True'
        if name=='MMMU_DEV_VAL':
            answer=answers[self.sample_id]
            return answer if len(answer)==1 and answer.isupper() else 'A'
        return 'Extracted answer: '+answers[self.sample_id]+'\nAnswer format: String'
    scoring.StrictJudge.generate=generate
    result=scoring.score(name,'mock',1)
    assert result['scoring_denominator']==2
    assert 0 <= result['accuracy_pct'] <= 100
    assert scoring.read(cell/'complete.json')['complete']
    duplicate=[predictions[0],predictions[0]]
    try:
        scoring.check_coverage(expected,duplicate)
    except ValueError:
        pass
    else:
        raise AssertionError('duplicate predictions accepted')
    empty=copy.deepcopy(predictions)
    empty[0]['prediction_text']=''
    try:
        scoring.check_coverage(expected,empty)
    except ValueError:
        pass
    else:
        raise AssertionError('empty predictions accepted')
    validation=dict(dataset=name,passed=True,accuracy_pct=result['accuracy_pct'],native_samples=2,
        rejection_checks=['duplicate','empty'],root=str(root))
    scoring.save(actual/'validation'/(name.lower()+'_native_cpu.json'),validation)
    print(json.dumps(validation))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('dataset')
    main(parser.parse_args().dataset)
