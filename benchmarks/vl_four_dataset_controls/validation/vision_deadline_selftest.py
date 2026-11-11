"""Exercise deadline rejection and owned-process cleanup without a model or GPU."""

from contextlib import contextmanager
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import vision_pipeline as pipeline
import tablemoe_common


def expect_timeout(call):
    try:
        call()
    except TimeoutError:
        return
    raise AssertionError('expired operation did not fail')


def main():
    evidence = ROOT / 'validation' / f'vision_deadline_{time.time_ns()}'
    evidence.mkdir()
    checks = []
    with patch.object(pipeline.time, 'time', return_value=100):
        assert pipeline.deadline_timeout(None, 1800) == 1800
        assert pipeline.deadline_timeout(10000, 1800) == 1800
        assert pipeline.deadline_timeout(100.25, 1800) == .25
        expect_timeout(lambda: pipeline.deadline_timeout(100, 1800))
    checks.append('positive_remaining_timeout_and_cap')

    with patch.object(pipeline, 'RUNS', evidence / 'expired'), patch.object(pipeline, 'config_for') as config:
        expect_timeout(lambda: pipeline.main(pipeline.DATASET, 'fused', 0, 'inference', True, time.time() - 1))
        config.assert_not_called()
        failure = pipeline.read(evidence / 'expired/smoke/fused/pipeline_inference_failure.json')
        assert failure['complete'] is False
    checks.append('expired_prepare_rejected_and_failure_saved')

    with patch.object(pipeline.subprocess, 'Popen') as popen:
        expect_timeout(lambda: pipeline.adapter_call('doctor', evidence / 'unused.json', evidence, 0, time.time() - 1))
        popen.assert_not_called()
    checks.append('expired_adapter_does_not_spawn')

    folder = evidence / 'startup/smoke/fused'
    folder.mkdir(parents=True)
    config = dict(output=str(folder), projects=['probe'], generation=dict(model='probe'))
    path = folder / 'run_config.json'
    pipeline.write_json(path, config)
    spawned = []
    timeouts = []
    real_popen = subprocess.Popen
    def tracked_popen(*args, **kwargs):
        proc = real_popen(*args, **kwargs)
        spawned.append(proc)
        return proc
    @contextmanager
    def bounded_server(args, server_dir, repo, env, timeout):
        timeouts.append(timeout)
        with tablemoe_common.running_server(args, server_dir, repo, env, timeout) as proc:
            yield proc
    command = [sys.executable, '-B', '-c', 'import time; time.sleep(30)']
    start = time.time()
    deadline = start + .25
    with patch.object(pipeline, 'RUNS', evidence / 'startup'), \
         patch.object(pipeline, 'prepare', return_value=(folder, path, config)), \
         patch.object(pipeline, 'stage_command', return_value=(command, 1)), \
         patch.object(pipeline, 'environment', return_value=dict(os.environ, CUDA_VISIBLE_DEVICES='')), \
         patch.object(pipeline, 'running_server', side_effect=bounded_server), \
         patch.object(tablemoe_common.subprocess, 'Popen', side_effect=tracked_popen):
        expect_timeout(lambda: pipeline.main(pipeline.DATASET, 'fused', 0, 'inference', True, deadline))
    assert len(spawned) == 1 and spawned[0].poll() is not None
    assert 0 < timeouts[0] <= .25 and time.time() - start < 5
    failure = pipeline.read(folder / 'pipeline_inference_failure.json')
    assert failure['complete'] is False and failure['deadline_unix'] == deadline
    checks.append('startup_timeout_capped_and_real_cpu_child_stopped')

    clock = [100]
    api_timeouts = []
    class MockOpener:
        def open(self, request, timeout):
            api_timeouts.append(timeout)
            clock[0] = 101
            return io.StringIO(json.dumps(dict(data=[dict(id='probe')])))
    with patch.object(pipeline.time, 'time', side_effect=lambda: clock[0]), \
         patch.object(pipeline.urllib.request, 'build_opener', return_value=MockOpener()):
        expect_timeout(lambda: pipeline.endpoint_probe(1, config, 'inference', folder, 100.5))
    assert api_timeouts == [.5]
    checks.append('api_timeout_capped_and_expiry_blocks_next_request')

    passed_deadlines = []
    def prepared_adapter(stage, path, folder, gpu, deadline):
        passed_deadlines.append((stage, deadline))
        if stage == 'doctor':
            raise TimeoutError('mock doctor deadline')
    with patch.object(pipeline, 'config_for', return_value=(folder, path, config)), \
         patch.object(pipeline, 'adapter_call', side_effect=prepared_adapter):
        future = time.time() + 30
        expect_timeout(lambda: pipeline.prepare('fused', 0, True, future))
    assert passed_deadlines == [('prepare', future), ('doctor', future)]
    checks.append('prepare_and_doctor_receive_same_deadline')

    for method in pipeline.METHODS:
        _, _, config = pipeline.config_for(method)
        for stage in ('inference', 'evaluate'):
            args, _ = pipeline.stage_command(method, 0, stage, config)
            assert args[args.index('--max-prefill-length') + 1] == '4096'
    checks.append('all_ten_commands_keep_prefill_4096')
    assert not (evidence / 'validation/vision_pipeline_ready.json').exists()
    result = dict(complete=True, checks=checks, gpu_started=False, real_model_called=False,
                  cpu_child_pid=spawned[0].pid, cpu_child_exit_code=spawned[0].returncode,
                  startup_timeout_s=timeouts[0], evidence_root=str(evidence))
    pipeline.write_json(ROOT / 'validation/vision_deadline_cpu.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
