import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import socket
import sys
import time

ROOT = Path(__file__).resolve().parent
SNAPSHOT = Path(os.environ.get('FREETOKEN_EXPERIMENT_SNAPSHOT',str(ROOT/'runtime_snapshot')))
PYTHON = '/home/fanya/local_archives/freetoken_consolidation_20261029_020126/preserved/FreeToken/.venv/bin/python'
MODEL = '/data/fanya/models/Qwen3-VL-30B-A3B-Instruct'
sys.path[:0] = [str(SNAPSHOT/'python'),str(SNAPSHOT/'benchmarks')]
from tablemoe_common import stream_request, running_server
from tablemoe_data import api_messages, read_rows
from freetoken.moe.tablemoe_artifacts import write_json


def environment(gpu):
    env = dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),CUDA_HOME='/usr/local/cuda-13.0',
        PYTHONPATH=f'{SNAPSHOT}/python:{SNAPSHOT}/benchmarks',HF_HUB_OFFLINE='1',
        HF_DATASETS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',
        PATH=f'{Path(PYTHON).parent}:/usr/local/cuda-13.0/bin:'+os.environ.get('PATH',''))
    for var,folder in [('TMPDIR','tmp'),('TMP','tmp'),('TEMP','tmp'),('PYTHONPYCACHEPREFIX','pycache'),
                       ('TRITON_CACHE_DIR','triton_cache'),('CUDA_CACHE_PATH','cuda_cache'),('TORCH_EXTENSIONS_DIR','torch_extensions')]:
        path=ROOT/folder
        path.mkdir(exist_ok=True)
        env[var]=str(path)
    return env


def processor(name):
    return dict(size=dict(shortest_edge=65536,longest_edge=1048576)) if name=='MMLongBench_DOC' else dict(size=dict(shortest_edge=602112,longest_edge=4014080))


def asset_root(name):
    if name=='MMMU_DEV_VAL':
        return Path('/data/fanya/tablemoe/mmmu_offline_bs16_seed42_split60_20261110_3_ks1wmc/build_bs16/assets')
    return Path('/data/fanya/tablemoe/vl_four_dataset_bs1_20261111')/name/'assets'


def command(name,method,gpu,collect=None):
    while True:
        with socket.socket() as probe, socket.socket() as backend:
            probe.bind(('127.0.0.1',0))
            port=probe.getsockname()[1]
            try:
                backend.bind(('127.0.0.1',port+1))
            except OSError:
                continue
            break
    strategy='fused' if method=='fused' else ('offload' if method=='modes' else 'tablemoe')
    cmd=[PYTHON,'-m','freetoken.cli','serve','--model',MODEL,'--gpu','0','--dtype','float16',
         '--moe-strategy',strategy,'--cuda-graph-max-bs','0' if collect else '1',
         '--max-running-requests','1','--num-pages','65536','--page-size','1','--cache-type','naive',
         '--mm-encoder-weights','gpu','--max-prefill-length','4096','--max-seq-len-override','65536',
         '--port',str(port),'--reasoning-parser','off','--mm-processor-kwargs',json.dumps(processor(name))]
    if method!='fused':
        cmd += ['--moe-cache-size','6144' if collect else '3072','--moe-cpu-threads','8']
    if method=='modes':
        cmd += ['--moe-policy','modes','--tablemoe-convert-bf16-weights','--modes-config',str(ROOT/'calibration'/name/'prefill_decode/modes_config/modes_config.json')]
    elif method!='fused':
        cmd += ['--tablemoe-implementation','optimized','--tablemoe-expert-mode','offload',
                '--tablemoe-convert-bf16-weights','--tablemoe-no-renormalize-weights',
                '--tablemoe-prefill-keep-rate','1' if collect else '.6',
                '--tablemoe-decode-top-h','8' if collect else '4',
                '--tablemoe-prefill-chunk-size','0','--tablemoe-prefill-experts-per-group','16',
                '--tablemoe-online-cache-size','128' if method=='tablemoe' and not collect else '0',
                '--tablemoe-lookup-mode',{'schedule':'none','table':'offline'}.get(method,'all')]
        if not collect and method!='schedule':
            assets=asset_root(name)
            cmd += ['--tablemoe-pca-dir',str(assets/'cluster'),'--tablemoe-cache-dir',str(assets/'cache'),
                    '--tablemoe-offline-cosine-threshold-vision','.6','--tablemoe-offline-cosine-threshold-text','.7',
                    '--tablemoe-decode-online-cosine-threshold','.7','--tablemoe-decode-offline-cosine-threshold','.7']
        if collect:
            cmd += ['--tablemoe-collect-config',str(collect)]
    return cmd,port


def main(name,method,gpu):
    folder=ROOT/'cells'/name/method
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/'generation_complete.json').exists():
        return
    if (folder/'requests.jsonl').exists():
        raise RuntimeError('partial cell needs fresh destination to replay cache order')
    rows=read_rows(ROOT/'prepare'/name/'evaluation.jsonl')
    calibration=read_rows(ROOT/'prepare'/name/'calibration.jsonl')
    cmd,port=command(name,method,gpu)
    limits={'MMMU_DEV_VAL':2048,'BabyVision':4096,'MMLongBench_DOC':2048,'Vision2Web_Level1':16384}
    env=environment(gpu)
    write_json(folder/'command.json',dict(command=cmd,gpu=gpu,
        environment={k:env[k] for k in ['CUDA_VISIBLE_DEVICES','PYTHONPATH','TMPDIR','OMP_NUM_THREADS']},batch_size=1))
    with running_server(cmd,folder/'server',SNAPSHOT,environment(gpu),1800):
        warmup=[]
        for row in calibration[:2]:
            warmup.append(stream_request(port,{**row,'messages':api_messages(row['vlmeval_message'])},16,natural=True))
        write_json(folder/'warmup.json',warmup)
        start=time.monotonic()
        with (folder/'requests.jsonl').open('w') as handle:
            for i,row in enumerate(rows):
                record=stream_request(port,{**row,'messages':api_messages(row['vlmeval_message'])},limits[name],natural=True,timeout=7200)
                handle.write(json.dumps(record,ensure_ascii=False)+'\n')
                handle.flush()
                write_json(folder/'progress.json',dict(completed=i+1,total=len(rows),elapsed_s=time.monotonic()-start,
                    updated_utc=datetime.now(timezone.utc).isoformat()))
    values=read_rows(folder/'requests.jsonl')
    assert [r['id'] for r in values]==[r['id'] for r in rows]
    wall=sum(r['complete_wall_s'] for r in values)
    tokens=sum(r['usage']['completion_tokens'] for r in values)
    tpots=[r['tpot_ms'] for r in values if r['tpot_ms'] is not None]
    write_json(folder/'generation_complete.json',dict(complete=True,samples=len(rows),returned_tokens=tokens,
        request_wall_s=wall,tokens_per_second=tokens/wall,mean_ttft_s=sum(r['ttft_s'] for r in values)/len(values),
        mean_tpot_ms=sum(tpots)/len(tpots) if tpots else None,
        length_capped=sum(r['finish_reason']=='length' for r in values),scoring_complete=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('dataset')
    parser.add_argument('method')
    parser.add_argument('--gpu',type=int,required=True)
    args=parser.parse_args()
    main(args.dataset,args.method,args.gpu)
