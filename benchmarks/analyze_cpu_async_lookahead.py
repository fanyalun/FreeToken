"""Measure useful ahead-token coverage and continuous useful lookahead distance."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics


THRESHOLDS = (16, 32, 64, 80, 96)


def effective_lookahead(rolling):
    output = rolling['output_tokens']
    returned = len(output)
    episodes, positions = [], set()
    distance = consumed = start = 0
    observed = False
    expected_mask = None
    for i, row in enumerate(rolling['handoffs']):
        proposed, accepted = row['proposed'], row['accepted']
        block = row['submitted']
        predictions = row['cpu_predictions']
        actual_accepted = next((j for j, pair in enumerate(zip(block['candidates'], predictions))
            if pair[0] != pair[1]), proposed)
        if not (len(block['candidates']) == proposed and len(predictions) == proposed + 1
                and accepted == actual_accepted and 0 <= accepted <= proposed
                and row['input_count'] == min(accepted + 1, returned - consumed - 1)
                and row['cpu_tokens'] == list(predictions[:row['input_count']])
                and list(row['cpu_tokens']) == list(output[consumed + 1:consumed + 1 + row['input_count']])):
            raise ValueError('CPU accepted-prefix accounting does not match saved tokens')
        mask = row.get('submitted_old_ahead_mask')
        if mask is None:
            if row['proposal_source'] not in ('initial', 'regenerated', 'replayed_ahead'):
                raise ValueError('legacy trace has unknown candidate provenance')
            mask = [row['proposal_source'] == 'replayed_ahead'] * proposed
        if len(mask) != proposed or any(type(value) is not bool for value in mask):
            raise ValueError('invalid ahead-token provenance mask')
        if expected_mask is not None and list(mask) != expected_mask:
            raise ValueError('recovery changed old-token provenance at the next CPU submission')
        useful = [consumed + 1 + j for j in range(accepted) if mask[j]]
        if positions.intersection(useful):
            raise ValueError('a returned ahead token was counted twice')
        positions.update(useful)
        count = len(useful)
        if row.get('cpu_accepted_old_ahead_tokens', count) != count:
            raise ValueError('saved useful-token count differs from the accepted provenance mask')
        distance += count
        consumed += row['input_count']
        # The first ahead token only supplies the CPU bonus boundary, not a reusable candidate.
        observed = observed or len(row['ahead']['candidates']) > 1 or any(mask)
        if row['terminal']:
            if i != len(rolling['handoffs']) - 1 or consumed != returned - 1:
                raise ValueError('terminal handoff does not finish the request')
            if observed:
                episodes.append(dict(start_handoff=start, end_handoff=i,
                    useful_tokens=distance, censored=True, reason='output_cap'))
            break
        reason = row['reason']
        capacity_stop = reason in ('no_ahead_capacity', 'no_retained_candidates')
        if capacity_stop and (not row['full_accept'] or returned - consumed - 1 != 1):
            raise ValueError('empty ahead block is not explained by the output budget')
        checked = row.get('recheck')
        if checked is not None:
            matched = checked['matched_old_tokens']
            size = len(row['next_proposal']['candidates'])
            if not 0 <= matched <= size:
                raise ValueError('recheck has an impossible matched prefix')
            expected_mask = [True] * matched + [False] * (size - matched)
            continue_old_ahead = matched > 0
        elif row['retained']:
            expected_mask = [True] * len(row['next_proposal']['candidates'])
            continue_old_ahead = bool(expected_mask)
        else:
            expected_mask = [False] * len(row['next_proposal']['candidates'])
            continue_old_ahead = False
        if not continue_old_ahead:
            if observed:
                episodes.append(dict(start_handoff=start, end_handoff=i,
                    useful_tokens=distance, censored=capacity_stop,
                    reason='output_cap' if capacity_stop else reason))
            distance, start, observed = 0, i + 1, False
    if consumed != returned - 1 or sum(e['useful_tokens'] for e in episodes) != len(positions):
        raise ValueError('useful-distance episodes do not conserve the accepted ahead tokens')
    return dict(episodes=episodes, useful_ahead_tokens=len(positions),
        output_tokens_excluding_initial_anchor=consumed,
        coverage=len(positions) / consumed if consumed else None,
        accepted_output_positions=sorted(positions))


def summarize_samples(samples):
    episodes = [e for sample in samples for e in sample['episodes']]
    values = [e['useful_tokens'] for e in episodes]
    n = len(episodes)
    survival = {}
    for threshold in THRESHOLDS:
        confirmed = sum(e['useful_tokens'] >= threshold for e in episodes)
        unknown = sum(e['censored'] and e['useful_tokens'] < threshold for e in episodes)
        survival[str(threshold)] = dict(confirmed=confirmed, early_censored=unknown, attempts=n,
            lower=confirmed / n if n else None, upper=(confirmed + unknown) / n if n else None)
    useful = sum(sample['useful_ahead_tokens'] for sample in samples)
    output = sum(sample['output_tokens_excluding_initial_anchor'] for sample in samples)
    return dict(samples=len(samples), attempts=n, useful_ahead_tokens=useful,
        output_tokens_excluding_initial_anchor=output, coverage=useful / output if output else None,
        zero_exact=sum(e['useful_tokens'] == 0 and not e['censored'] for e in episodes),
        zero_censored=sum(e['useful_tokens'] == 0 and e['censored'] for e in episodes),
        output_censored=sum(e['censored'] for e in episodes),
        observed_median_lower_bound=statistics.median(values) if values else None,
        histogram_observed=dict(sorted(Counter(values).items())), survival=survival,
        initial_episodes=[dict(sample=sample['sample'], **sample['episodes'][0])
            for sample in samples if sample['episodes']], sample_metrics=samples)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def survival_text(row):
    if row['lower'] is None:
        return '待测'
    lower, upper = row['lower'] * 100, row['upper'] * 100
    return f'{lower:.1f}%' if lower == upper else f'{lower:.1f}%–{upper:.1f}%'


def analyze(manifest, *, allow_partial=False):
    cases, inputs = [], []
    for case in manifest['cases']:
        directory = Path(case['path'])
        samples = []
        for sample in range(case['samples']):
            root = directory / f'sample_{sample:02d}'
            marker = root / 'complete.json'
            if not marker.exists() or read(marker).get('status') != 'complete':
                if allow_partial:
                    continue
                raise ValueError(f'incomplete sample: {root}')
            path = root / 'rolling.json'
            fingerprint = digest(path)
            raw = read(path)
            if len(raw['output_tokens']) != case['tokens'] or not read(marker)['valid']:
                raise ValueError(f'invalid or wrong-length sample: {root}')
            measured = effective_lookahead(raw)
            measured.update(sample=sample)
            samples.append(measured)
            inputs.append(dict(path=str(path.resolve()), sha256=fingerprint))
        summary = summarize_samples(samples)
        summary.update(id=case['id'], model=case['model'], variant=case['variant'],
            source=str(directory), expected_samples=case['samples'],
            complete=len(samples) == case['samples'])
        cases.append(summary)
    for source in inputs:
        if digest(Path(source['path'])) != source['sha256']:
            raise ValueError('a source trace changed during analysis')
    return dict(metric_version='effective_lookahead_v1',
        complete=all(c['complete'] for c in cases), algorithm_only=True, performance='not_measured',
        definition=dict(distance='CPU-accepted old-ahead candidates until no old ahead prefix remains usable',
            partial_recheck='positive matched prefix continues; zero match restarts',
            coverage='unique CPU-accepted old-ahead candidate positions / outputs excluding initial anchor',
            exclusions='CPU bonus, initial anchor, corrections and fresh regeneration are not old-ahead candidates',
            censor='output cap ends observation; threshold probability lies between lower and upper bounds',
            survival_bounds='confirmed reaches / all starts; upper adds output-censored starts below the threshold',
            thresholds=list(THRESHOLDS)), cases=cases, source_traces=inputs,
        analysis_source=dict(path=str(Path(__file__).resolve()), sha256=digest(Path(__file__))))


def report(result):
    lines = ['# CPU 异步验证：有效前瞻距离与覆盖率', '',
        'D 是一次启动后，直到旧前瞻完全失效为止，最终由 CPU 接受的旧候选 token 总数。重查保留正长度旧前缀时继续累计，修改尾部不自动断链；零匹配或 CPU/边界拒绝才重新启动。', '',
        '覆盖率为 CPU 接受的旧候选 token 数 / 返回 token 数（排除每条的初始 anchor）。不计 CPU bonus、重查新产生的修正 token 或收到 CPU 结果后新生成的 token；同一输出位置只计一次。', '',
        '16/32/64/80/96 是固定 token 距离，方便跨配置比较；80/96 相当于五/六个 16-token 观察单位，不冒充实际五/六个可变长 GPU 块。', '',
        '区间下界只计已确认到达阈值的尝试；上界把未到阈值就达到输出上限的尝试视为可能继续到达。不把输出上限停止算作算法失败，不对未观察的后续作独立性假设。', '',
        '| 模型 / 配置 | 样本 | 有效前瞻覆盖率 | 尝试数 | 确认 D=0 | D≥16 | D≥32 | D≥64 | D≥80 | D≥96 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for c in result['cases']:
        coverage = f"{100*c['coverage']:.2f}% ({c['useful_ahead_tokens']}/{c['output_tokens_excluding_initial_anchor']})" if c['coverage'] is not None else '待测'
        tail = ' | '.join(survival_text(c['survival'][str(t)]) for t in THRESHOLDS)
        lines.append(f"| {c['model']} / {c['variant']} | {c['samples']}/{c['expected_samples']} | {coverage} | {c['attempts']} | {c['zero_exact']} | {tail} |")
    lines += ['', '同一请求中的重启不是独立样本。观察到的距离中位数及完整逐次记录保存在 metrics.json；达到输出上限的 D 是下界。', '',
        '本次仅衡量算法可利用性，不测性能，不忽略 CPU 数值与状态正确性检查。', '',
        '[结构化指标、逐次距离和来源 SHA-256](metrics.json) · [复算校验](validation.json)']
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--allow-partial', action='store_true')
    args = parser.parse_args()
    out = args.output.resolve()
    if out == Path('/tmp') or Path('/tmp') in out.parents:
        parser.error('use persistent output outside /tmp')
    result = analyze(read(args.manifest), allow_partial=args.allow_partial)
    out.mkdir(parents=True, exist_ok=True)
    for name, value in [('metrics.json', result), ('validation.json', dict(valid=True,
            complete=result['complete'], source_traces=len(result['source_traces']),
            accepted_token_conservation=True, provenance_and_cpu_prefix_checks=True,
            sources_unchanged=True, new_gpu_inference=False))]:
        stage = out / (name + '.new')
        stage.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
        stage.replace(out / name)
    (out / 'readme.md').write_text(report(result))
    print(json.dumps(dict(complete=result['complete'], samples=sum(c['samples'] for c in result['cases']))))


if __name__ == '__main__':
    main()
