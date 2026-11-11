import csv
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
DATASETS=['Vision2Web_Level1','MMLongBench_DOC','BabyVision','MMMU_DEV_VAL']
METHODS=['fused','tablemoe','modes','schedule','table']


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def main():
    rows=[]
    for dataset in DATASETS:
        split=read(ROOT/'prepare'/dataset/'split.json')
        for method in METHODS:
            folder=ROOT/'cells'/dataset/method
            generation=read(folder/'generation_complete.json')
            complete=read(folder/'complete.json')
            score_path=Path(complete.get('score_file',folder/'score_local_fused_native/score.json'))
            score=read(score_path)
            progress=read(folder/'progress.json')
            bounded_folder=folder/'score_local_fused_native_bounded_2048'
            bounded_marker=read(bounded_folder/'traversal_complete.json')
            bounded=read(bounded_folder/'score.json') or read(bounded_folder/'progress.json')
            row=dict(dataset=dataset,method=method,batch_size=1,seed=42,
                evaluation_samples=split.get('evaluation_samples'),
                generation_complete=bool(generation.get('complete')),scoring_complete=bool(complete.get('complete')),
                completed_samples=generation.get('samples',progress.get('completed',0)),
                returned_tokens=generation.get('returned_tokens'),request_wall_s=generation.get('request_wall_s'),
                tokens_per_second=generation.get('tokens_per_second'),mean_ttft_s=generation.get('mean_ttft_s'),
                mean_tpot_ms=generation.get('mean_tpot_ms'),length_capped=generation.get('length_capped'),
                accuracy_pct=score.get('accuracy_pct'),
                visual_score_pct=complete.get('visual_score_pct',complete.get('visual_score')),
                score_source=str(score_path) if score else None,
                bounded_judging_complete=bool(bounded_marker.get('traversal_complete')),
                bounded_known_count=bounded.get('known_count'),bounded_unknown=bounded.get('unknown'),
                bounded_pending=bounded.get('pending'),
                accuracy_lower_pct=bounded.get('full_denominator_accuracy_lower_pct'),
                accuracy_upper_pct=bounded.get('full_denominator_accuracy_upper_pct'),
                bounded_score_source=str(bounded_folder) if bounded else None,
                source_folder=str(folder),score_details=score if score else None)
            rows.append(row)
    result=dict(complete=all(r['generation_complete'] and r['scoring_complete'] for r in rows),
        expected_cells=20,completed_generation_cells=sum(r['generation_complete'] for r in rows),
        completed_scored_cells=sum(r['scoring_complete'] for r in rows),
        completed_bounded_judging_cells=sum(r['bounded_judging_complete'] for r in rows),
        execution=read(ROOT/'execution_policy.json'),status=read(ROOT/'status.json'),rows=rows,
        tables={d:read(ROOT/'build'/d/'complete.json') for d in DATASETS if d!='MMMU_DEV_VAL'})
    temporary=ROOT/'summary.json.partial'
    temporary.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    temporary.replace(ROOT/'summary.json')
    fields=[k for k in rows[0] if k!='score_details']
    with (ROOT/'summary.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore')
        writer.writeheader();writer.writerows(rows)


if __name__=='__main__':main()
