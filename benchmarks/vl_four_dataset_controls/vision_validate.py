"""CPU checks for the official-workflow commands and frozen project selection."""

import ast
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from unittest.mock import patch

import vision_pipeline as pipeline
from freetoken.server.args import parse_args


def main():
    checks = []
    root = pipeline.ROOT
    ast.parse((root / 'vision_pipeline.py').read_text())
    with (root / 'logs/vision_command_validation.log').open('w') as log, redirect_stdout(log), redirect_stderr(log):
        for method in pipeline.METHODS:
            folder, path, config = pipeline.config_for(method)
            for stage in ('inference', 'evaluate'):
                cmd, port = pipeline.stage_command(method, 0, stage, config)
                args, _ = parse_args(cmd[cmd.index('serve') + 1:])
                assert args.max_seq_len_override == 200000
                assert args.num_page_override == 200000
                assert args.max_running_req == 1
                assert args.max_output_tokens == 64000
                strategy = cmd[cmd.index('--moe-strategy') + 1]
                expected = 'fused' if stage == 'evaluate' or method == 'fused' else 'offload' if method == 'modes' else 'tablemoe'
                assert strategy == expected
                assert args.reasoning_parser is None
                assert args.tool_call_parser == 'qwen25'
                if stage == 'inference' and method in ('tablemoe', 'schedule', 'table'):
                    assert args.tablemoe_prefill_keep_rate == .6
                    assert args.tablemoe_decode_keep_k == 4 and args.moe_cache_size == 3072
                    assert args.tablemoe_lookup_mode == dict(tablemoe='all', schedule='none', table='offline')[method]
                    assert not args.tablemoe_renormalize_weights
                checks.append(dict(method=method, stage=stage, strategy=strategy, context=200000, bs=1, port=port))
    calibration = {row['metadata']['task'] for row in pipeline.read_rows(root / 'prepare/Vision2Web_Level1/calibration.jsonl')}
    assert len(config['projects']) == 40 and not calibration & set(config['projects'])
    key = f'webpage:claude_code:{config["generation"]["model"]}'
    statistics = dict(total_count=40, prototype_scores=[.5] * 120,
                      desktop_scores=[.6] * 40, tablet_scores=[.5] * 40, mobile_scores=[.4] * 40)
    def score_records(path):
        return {'complete': True} if Path(path).name == 'complete.json' else {'statistics': {key: statistics}}
    with patch.object(pipeline, 'read', side_effect=score_records):
        result = pipeline.score_summary(config, folder)
        assert result['visual_score_pct'] == 50 and result['scoring_denominator_viewports'] == 120
        statistics['prototype_scores'].pop()
        try:
            pipeline.score_summary(config, folder)
        except ValueError:
            pass
        else:
            raise AssertionError('missing viewport scores must fail summary')
    output = root / 'validation/vision_pipeline_cpu.json'
    pipeline.write_json(output, dict(complete=True, command_checks=checks, syntax=True, split_disjoint=True,
        score_denominator_checks=2, gpu_started=False, api_probe_performed=False, official_inference_performed=False))
    print(output)


if __name__ == '__main__':
    main()
