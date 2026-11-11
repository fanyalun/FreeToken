import json
import math
import os
from pathlib import Path
import sys
import time

if __name__=='__main__' and sys.argv[1] in ('MMLongBench_DOC','Vision2Web_Level1'):
    os.environ['FREETOKEN_EXPERIMENT_SNAPSHOT']=str(Path(__file__).resolve().parent/'runtime_snapshot_compact')

from run_cell import ROOT,SNAPSHOT,MODEL,asset_root,command,environment,processor
from tablemoe_common import running_server,stream_request
from tablemoe_data import read_rows,api_messages
from freetoken.moe.tablemoe_artifacts import read_json,write_json,digest,verify_stage
from freetoken.moe.tablemoe_builder import algorithm_config,finalize_collection,cluster,materialize


def main(name,gpu):
    os.environ['CUDA_VISIBLE_DEVICES']=str(gpu)
    rows=read_rows(ROOT/'prepare'/name/'calibration.jsonl')
    assets=asset_root(name)
    assets.mkdir(parents=True,exist_ok=True)
    folder=ROOT/'build'/name
    folder.mkdir(parents=True,exist_ok=True)
    algorithm=algorithm_config(dict(seed=42,clusters=256,pca_dim=64,reservoir_size=30000,
        pca_per_expert=1000,iterations=30,cluster_batch_size=8,distance_chunk=2048,value_chunk=64,
        device='cuda',dtype='float16',export_inputs=False,sampling_policy='tablemoe_default',
        sample_count=len(rows),sample_decode_tokens=127,checkpoint_interval=1,per_sample_seed=True,
        min_vision_per_sample=1,max_vision_per_sample=1000,min_text_per_sample=8,max_text_per_sample=500,
        max_vision_tokens=50000,max_text_tokens=40000,collection_image_ids=[r['image_ids'] for r in rows],
        **(dict(raw_storage='indexed') if name in ('MMLongBench_DOC','Vision2Web_Level1') else {})))
    checkpoint=read_json(ROOT.parents[0]/'tablemoe_sensitivity_preserve_vlmeval_20261110/checkpoint_binding.json')
    inputs=dict(checkpoint=checkpoint,convert_bf16=True,processor=processor(name),
        calibration_sha256=digest(rows),calibration_ids=[r['id'] for r in rows],sample_count=len(rows),algorithm=algorithm)
    collect=assets/'collect'
    cfg=assets/'collect_config.json'
    write_json(cfg,dict(output=str(collect),builder=algorithm,input_sha256=digest(inputs)))
    write_json(folder/'protocol.json',dict(inputs=inputs,assets=str(assets),batch_size=1,
        full_multimodal_input_preserved=True,shared_visual_collection_deduplicated=True,
        runtime_snapshot=str(SNAPSHOT),raw_storage=algorithm.get('raw_storage','per_expert')))
    def stage(value):
        write_json(folder/'status.json',dict(stage=value,updated_unix=time.time(),complete=False))
        print(name,value,flush=True)
    if not (collect/'complete.json').exists():
        if (collect/'progress.json').exists():
            raise RuntimeError('partial collection needs verified resume before relaunch')
        cmd,port=command(name,'tablemoe',gpu,collect=cfg)
        stage('collection')
        write_json(folder/'command.json',dict(command=cmd,gpu=gpu))
        with running_server(cmd,folder/'server',SNAPSHOT,environment(gpu),1800):
            collect.mkdir(exist_ok=True)
            with (collect/'requests.jsonl').open('w') as handle:
                for i,row in enumerate(rows):
                    value=stream_request(port,{**row,'messages':api_messages(row['vlmeval_message'])},128,natural=False,timeout=7200)
                    handle.write(json.dumps(value,ensure_ascii=False)+'\n')
                    handle.flush()
                    write_json(folder/'progress.json',dict(completed=i+1,total=len(rows),stage='collection'))
        stage('finalize_collection')
        finalize_collection(collect,inputs,48,128)
    for label,fn in [('cluster',cluster),('materialize',materialize)]:
        stage(label)
        fn(assets,algorithm)
    manifest=verify_stage(assets/'cache',hash_outputs=False)
    progress=read_json(collect/'progress.json')
    records=progress['sample_state']['records']
    write_json(folder/'complete.json',dict(complete=True,assets=str(assets),calibration_samples=len(rows),
        missing_tables=manifest['missing_tables'],collected_tokens=progress['sample_state']['counts'],
        first_seen_images=sum(len(r['new_image_ids']) for r in records),
        image_count_semantics='first encountered images; token sampling may select zero positions for an image',
        duplicate_visual_tokens=sum(r['duplicate_visual_tokens'] for r in records),tensor_content_scans=False))
    write_json(folder/'status.json',dict(stage='complete',complete=True,updated_unix=time.time()))


if __name__=='__main__':
    main(sys.argv[1],int(sys.argv[2]))
