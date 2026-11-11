import argparse
import json
from pathlib import Path
import subprocess
import signal
import time

from run_cell import ROOT,SNAPSHOT,command,environment
from tablemoe_common import running_server
from freetoken.moe.tablemoe_artifacts import write_json
from score_native import score_options


def main(dataset, methods, gpu, *, judge_max_tokens=None, score_directory=None):
    maximum, directory = score_options(dataset, judge_max_tokens, score_directory)
    def complete(method):
        folder=ROOT/'cells'/dataset/method/directory
        path=folder/'complete.json'
        if not path.exists() or not json.loads(path.read_text()).get('complete'):
            return False
        protocol=json.loads((folder/'score.json').read_text())['protocol']
        if (protocol.get('judge_max_tokens') != maximum
                or protocol.get('score_directory', 'score_local_fused_native') != directory):
            raise ValueError('completed score uses a different judge budget or directory')
        return True
    methods = [method for method in methods if not complete(method)]
    if not methods:
        return
    for method in methods:
        marker=ROOT/'cells'/dataset/method/'generation_complete.json'
        if not marker.exists() or not json.loads(marker.read_text()).get('complete'):
            raise ValueError(f'generation incomplete for {dataset}/{method}')
    folder=ROOT/'judges'/dataset/str(time.time_ns())
    cmd,port=command(dataset,'fused',gpu)
    env=environment(gpu)
    write_json(folder/'command.json',dict(command=cmd,gpu=gpu,batch_size=1,
        judge_dtype='float16',judge_moe_strategy='fused',methods=methods,
        judge_max_tokens=maximum,score_directory=directory))
    with running_server(cmd,folder/'server',SNAPSHOT,env,1800):
        for method in methods:
            kit_env={**env,'CUDA_VISIBLE_DEVICES':'', 'LD_LIBRARY_PATH':'/home/fanya/miniforge/envs/tablemoe/lib'}
            command_args=['/home/fanya/miniforge/envs/tablemoe/bin/python','-B',str(ROOT/'score_native.py'),
                          dataset,method,'--port',str(port),'--judge-max-tokens',str(maximum),
                          '--score-directory',directory]
            with (folder/(method+'.log')).open('w') as log:
                subprocess.run(command_args,cwd=ROOT,env=kit_env,stdout=log,stderr=subprocess.STDOUT,check=True)
    write_json(folder/'complete.json',dict(complete=True,dataset=dataset,methods=methods,
        judge_max_tokens=maximum,score_directory=directory))


if __name__=='__main__':
    def interrupted(signum,frame):
        raise KeyboardInterrupt(f'judge wrapper interrupted by signal {signum}')
    signal.signal(signal.SIGTERM,interrupted)
    signal.signal(signal.SIGINT,interrupted)
    parser=argparse.ArgumentParser()
    parser.add_argument('dataset')
    parser.add_argument('--methods',nargs='+',required=True)
    parser.add_argument('--gpu',type=int,required=True)
    parser.add_argument('--judge-max-tokens',type=int)
    parser.add_argument('--score-directory')
    args=parser.parse_args()
    main(args.dataset,args.methods,args.gpu,judge_max_tokens=args.judge_max_tokens,
         score_directory=args.score_directory)
