"""Own one fused BS1 judge for independent Doc bounded512 continuation."""

import argparse
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import time

from run_cell import ROOT, SNAPSHOT, command, environment
from tablemoe_common import running_server
from freetoken.moe.tablemoe_artifacts import write_json
from score_doc_bounded import DATASET, DIRECTORY
from score_native import fingerprint


def main(method, gpu):
    destination = ROOT / 'cells' / DATASET / method / DIRECTORY
    marker = destination / 'traversal_complete.json'
    if marker.exists():
        value = json.loads(marker.read_text())
        if value.get('complete') and value.get('traversal_complete') and value.get('sample_count') == 436:
            score = json.loads((destination / 'score.json').read_text())
            records = json.loads((destination / 'records.json').read_text())
            if (score['processed'] != 436 or len(records) != 436 or score['strict_primary_complete'] is not False
                    or value.get('strict_primary_complete') is not False
                    or score['protocol']['judge_max_tokens'] != 512
                    or score['protocol']['score_directory'] != DIRECTORY
                    or value.get('records_sha256') != fingerprint(records)
                    or value.get('protocol_sha256') != fingerprint(score['protocol'])):
                raise ValueError('bounded Doc completion has invalid coverage or primary status')
            return
    generation = ROOT / 'cells' / DATASET / method / 'generation_complete.json'
    if not json.loads(generation.read_text()).get('complete'):
        raise ValueError('generation is incomplete')
    folder = ROOT / 'judges' / DATASET / ('bounded_' + str(time.time_ns()))
    cmd, port = command(DATASET, 'fused', gpu)
    env = environment(gpu)
    worker = ['/home/fanya/miniforge/envs/tablemoe/bin/python', '-B', str(ROOT / 'score_doc_bounded.py'),
              method, '--port', str(port)]
    write_json(folder / 'command.json', dict(command=cmd, worker=worker, gpu=gpu, batch_size=1,
        judge_dtype='float16', judge_moe_strategy='fused', judge_max_tokens=512,
        score_directory=DIRECTORY, strict_primary_complete=False,
        wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    with running_server(cmd, folder / 'server', SNAPSHOT, env, 1800):
        kit_env = {**env, 'CUDA_VISIBLE_DEVICES': '', 'LD_LIBRARY_PATH': '/home/fanya/miniforge/envs/tablemoe/lib'}
        with (folder / (method + '.log')).open('w') as log:
            subprocess.run(worker, cwd=ROOT, env=kit_env, stdout=log, stderr=subprocess.STDOUT, check=True)
    write_json(folder / 'traversal_complete.json', dict(complete=True, method=method, strict_primary_complete=False,
        score_marker=str(marker)))


if __name__ == '__main__':
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'bounded Doc judge interrupted by signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('method', choices=('fused', 'tablemoe', 'modes', 'schedule', 'table'))
    parser.add_argument('--gpu', type=int, required=True)
    args = parser.parse_args()
    main(args.method, args.gpu)
