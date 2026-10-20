"""Real-model CPU worker rejection, cancellation and stale-epoch audit."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch

from bench_cpu_verify_feasibility import choose_cores, wait_for
from freetoken.engine.qwen_cpu_target import PrefixCheckpoint, VerifyJob
from freetoken.engine.qwen_cpu_worker import CpuVerifierWorker


def digest(tensor):
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def check_result(result, expected, token, length):
    assert result.tokens == (token,) and result.checkpoint.anchor == token
    assert result.checkpoint.length == length and result.start == length - 1
    values = dict(hidden=result.hidden, predictions=result.predictions[:1],
        **{field: getattr(result.checkpoint, field) for field in ('keys', 'values', 'recurrent', 'conv')})
    actual = {key: digest(tensor) for key, tensor in values.items()}
    if actual != expected:
        raise RuntimeError(f'worker result differs from serial reference: {actual} != {expected}')


def audit_widths(worker, checkpoint, reference, widths, epoch, vocab):
    checks = []
    for width in widths:
        rejections = [None] if width == 1 else list(dict.fromkeys([None, 0, (width - 2) // 2]))
        for reject in rejections:
            epoch += 1
            worker.initialize(checkpoint, epoch)
            wait_for(worker, 'initialized')
            candidates = list(reference['tokens'][:width - 1])
            if reject is not None:
                candidates[reject] = (candidates[reject] + 1) % vocab
            worker.submit(VerifyJob(epoch, 0, checkpoint.length, checkpoint.anchor, tuple(candidates), 128))
            result, _ = wait_for(worker, 'result')
            count = width if reject is None else reject + 1
            assert result.accepted == count - 1 and result.input_count == count
            assert result.start == checkpoint.length and result.checkpoint.length == checkpoint.length + count
            assert result.tokens == tuple(reference['tokens'][:count])
            assert result.checkpoint.anchor == result.tokens[-1]
            for row in range(count):
                expected = reference['serial_tensor_sha256'][row]
                assert digest(result.hidden[row:row + 1]) == expected['hidden']
                assert digest(result.predictions[row:row + 1]) == expected['predictions']
                for field in ('keys', 'values'):
                    assert digest(getattr(result.checkpoint, field)[:, row:row + 1]) == expected[field]
            for field in ('recurrent', 'conv'):
                assert digest(getattr(result.checkpoint, field)) == reference['serial_tensor_sha256'][count - 1][field]
            length, anchor = result.checkpoint.length, result.checkpoint.anchor
            worker.release(result)
            worker.submit(VerifyJob(epoch, 1, length, anchor, (), 128))
            following, _ = wait_for(worker, 'result')
            check_result(following, reference['serial_tensor_sha256'][count],
                int(following.predictions[0]), length + 1)
            worker.release(following)
            checks.append(dict(width=width, reject_at=reject, input_count=count, next_step_exact=True))
    return epoch, checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix-run', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--threads', type=int, required=True)
    parser.add_argument('--components', nargs='+', required=True)
    parser.add_argument('--widths', type=int, nargs='+', default=[])
    args = parser.parse_args()
    if any(not 1 <= width <= 33 for width in args.widths) or len(set(args.widths)) != len(args.widths):
        parser.error('widths must be unique values in [1,33]')
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    cores = choose_cores(args.threads)
    os.sched_setaffinity(0, {0, 24})
    identity = json.loads((args.prefix_run / 'config.json').read_text())
    config = json.loads((Path(identity['model']) / 'config.json').read_text())['text_config']
    references = {r['sample_id']: r for r in json.loads((args.reference / 'correctness.json').read_text())}
    if set(references) != set(range(identity['samples'])):
        raise ValueError('complete serial references required')
    worker = CpuVerifierWorker(identity['model'], config, cores, max_context=4300, max_rows=33,
        kernel_profile='ggml_v2', components=args.components, bind_openmp=True)
    checks = []
    epoch = 0
    try:
        worker.wait_ready()
        (args.output / 'runtime.json').write_text(json.dumps(worker.metadata, indent=2))
        for sample, reference in sorted(references.items()):
            checkpoint = PrefixCheckpoint(**torch.load(args.prefix_run / f'gpu_prefix_sample{sample}.pt',
                                                       map_location='cpu', weights_only=True))
            epoch, width_checks = audit_widths(worker, checkpoint, reference, args.widths, epoch, config['vocab_size'])
            epoch += 1
            worker.initialize(checkpoint, epoch)
            wait_for(worker, 'initialized')
            length, anchor = checkpoint.length, checkpoint.anchor
            for block in range(3):
                wrong = (reference['tokens'][block] + 1) % config['vocab_size']
                worker.submit(VerifyJob(epoch, block, length, anchor, (wrong,), 128))
                result, _ = wait_for(worker, 'result')
                assert result.accepted == 0 and result.input_count == 1
                assert result.tokens == (reference['tokens'][block],)
                check_result(result, reference['serial_tensor_sha256'][block], reference['tokens'][block], length + 1)
                length, anchor = result.checkpoint.length, result.checkpoint.anchor
                worker.release(result)
            try:
                worker.submit(VerifyJob(epoch - 1, 3, length, anchor, (), 128))
            except RuntimeError:
                pass
            else:
                raise AssertionError('stale epoch was accepted')
            worker.submit(VerifyJob(epoch, 3, length, anchor, (), 128))
            worker.cancel(epoch)
            wait_for(worker, 'cancelled')
            assert worker.inflight is None and all(owner is None for owner in worker.owners)
            epoch += 1
            worker.initialize(checkpoint, epoch)
            wait_for(worker, 'initialized')
            worker.submit(VerifyJob(epoch, 0, checkpoint.length, checkpoint.anchor,
                tuple(reference['tokens'][:3]), 1))
            result, _ = wait_for(worker, 'result')
            assert result.terminal and result.input_count == 1 and result.accepted == 3
            check_result(result, reference['serial_tensor_sha256'][0], reference['tokens'][0], checkpoint.length + 1)
            worker.release(result)
            epoch += 1
            worker.initialize(checkpoint, epoch)
            wait_for(worker, 'initialized')
            # An explicit stop ID exercises EOS handling without claiming a natural EOS.
            worker.submit(VerifyJob(epoch, 0, checkpoint.length, checkpoint.anchor,
                tuple(reference['tokens'][:3]), 128, eos_ids=(reference['tokens'][0],)))
            result, _ = wait_for(worker, 'result')
            assert result.terminal and result.input_count == 1 and result.accepted == 3
            check_result(result, reference['serial_tensor_sha256'][0], reference['tokens'][0], checkpoint.length + 1)
            worker.release(result)
            checks.append(dict(sample_id=sample, width_checks=width_checks, consecutive_rejections=3,
                exact_state=True, cancel_inflight=True, stale_epoch_rejected=True, repeated_request_exact=True,
                output_limit_checked=True, configured_stop_token_checked=True))
            (args.output / 'checks.json').write_text(json.dumps(checks, indent=2))
            print(json.dumps(checks[-1]), flush=True)
        worker.submit(VerifyJob(epoch, 1, -1, anchor, (), 128))
        try:
            wait_for(worker, 'result')
        except RuntimeError as exc:
            if 'CPU verify job does not start' not in str(exc):
                raise
            (args.output / 'expected_worker_exception.txt').write_text(str(exc))
        else:
            raise AssertionError('malformed job was accepted')
        assert not torch.cuda.is_initialized()
        (args.output / 'complete.json').write_text(json.dumps(dict(validity='valid', samples=len(checks),
            audited_widths=args.widths,
            real_model_worker_lifecycle=True, exception_reported=True, parent_cuda_initialized=False), indent=2))
    finally:
        worker.close()


if __name__ == '__main__':
    main()
