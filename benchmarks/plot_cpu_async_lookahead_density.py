"""Plot observed useful-distance densities with a separate zero-probability atom."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import numpy as np


METHODS = [
    ('top4_replay', 'Top-4 + replay', '#4C78A8', '--'),
    ('top4_recheck', 'Top-4 + recheck', '#F2B447', '-'),
    ('top6_replay', 'Top-6 + replay', '#59A14F', '--'),
    ('top6_recheck', 'Top-6 + recheck', '#E45756', '-'),
]


def reflected_density(x, positive, total, bandwidth):
    values = np.asarray(positive, dtype=float)
    if not len(values):
        return np.zeros_like(x)
    left = (x[:, None] - values[None, :]) / bandwidth
    right = (x[:, None] + values[None, :]) / bandwidth
    kernels = np.exp(-left * left / 2) + np.exp(-right * right / 2)
    return kernels.sum(axis=1) / (total * bandwidth * np.sqrt(2 * np.pi))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metrics', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--validation', type=Path, required=True)
    parser.add_argument('--bandwidth', type=float, default=8)
    args = parser.parse_args()
    for path in (args.output, args.validation.parent):
        resolved = path.resolve()
        if resolved == Path('/tmp') or Path('/tmp') in resolved.parents:
            parser.error('use persistent output outside /tmp')
        path.mkdir(parents=True, exist_ok=True)
    if args.bandwidth <= 0:
        parser.error('bandwidth must be positive')
    source = args.metrics.read_bytes()
    metrics = json.loads(source)
    assert metrics['complete'] and len(metrics['cases']) == 8
    cases = {c['id']: c for c in metrics['cases']}
    x = np.linspace(0, 320, 5121)
    curves, receipt = {}, []
    for model in ('qwen36', 'qwen235'):
        for variant, label, color, style in METHODS:
            case = cases[model + '_' + variant]
            episodes = [e for sample in case['sample_metrics'] for e in sample['episodes']]
            values = [e['useful_tokens'] for e in episodes]
            n = len(values)
            assert n == case['attempts'] and sum(v == 0 for v in values) == case['zero_exact']
            assert case['zero_censored'] == 0 and case['samples'] == 10
            density = reflected_density(x, [v for v in values if v > 0], n, args.bandwidth)
            p0 = sum(v == 0 for v in values) / n
            area = float(((density[1:] + density[:-1]) * np.diff(x) / 2).sum())
            assert abs(p0 + area - 1) < 1e-5
            censored = [e['useful_tokens'] for e in episodes if e['censored']]
            assert len(censored) == case['output_censored']
            curves[model + '_' + variant] = (density, p0, censored)
            receipt.append(dict(case=case['id'], attempts=n, zero_probability=p0,
                positive_density_area=area, total_probability=p0 + area,
                output_censored=len(censored), observed_distances=values))
    plt.rcParams.update({
        'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix',
        'font.size': 22, 'axes.labelsize': 22, 'xtick.labelsize': 20,
        'ytick.labelsize': 20, 'legend.fontsize': 20, 'axes.linewidth': 1,
        'pdf.fonttype': 42, 'ps.fonttype': 42,
    })
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.8), sharex=True, sharey=True)
    fig.subplots_adjust(left=.075, right=.995, bottom=.24, top=.80, wspace=.12)
    ymax = max(curve[0].max() for curve in curves.values()) * 1.12
    for ax, model, caption in zip(axes, ('qwen36', 'qwen235'), ('(a) 35B', '(b) 235B')):
        inset = ax.inset_axes([.55, .40, .43, .55])
        for i, (variant, label, color, style) in enumerate(METHODS):
            density, p0, censored = curves[model + '_' + variant]
            ax.plot(x, density, color=color, linestyle=style, linewidth=2.6, label=label)
            ax.plot(censored, np.interp(censored, x, density), linestyle='none', marker='v',
                markersize=7.5, markerfacecolor='white', markeredgecolor=color,
                markeredgewidth=1.4, zorder=4)
            inset.bar(i, p0, color=color, width=.65)
        inset.set_title(r'$P(D=0)$', fontsize=21, pad=4)
        inset.set_xticks(range(4), ['4R', '4C', '6R', '6C'], fontsize=18)
        inset.set_ylim(0, .55)
        inset.set_yticks([0, .25, .5])
        inset.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        inset.tick_params(axis='y', labelsize=17, length=3)
        inset.tick_params(axis='x', length=0, pad=4)
        inset.spines[['top', 'right']].set_visible(False)
        inset.spines[['bottom', 'left']].set_color('#999999')
        inset.set_facecolor('white')
        ax.set_xlim(0, 256)
        ax.set_ylim(0, ymax)
        ax.set_xticks([0, 64, 128, 192, 256])
        ax.set_xlabel(r'Observed useful lookahead $D$ (tokens)', labelpad=9)
        ax.text(.5, -.30, caption, transform=ax.transAxes, ha='center', va='center', fontsize=23)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', color='#E6E6E6', linewidth=.8)
        ax.set_axisbelow(True)
    axes[0].set_ylabel('Probability density (per token)', labelpad=10)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.53, 1.015),
        ncol=4, frameon=False, handlelength=2.4, columnspacing=1.2)
    fig.legend([Line2D([], [], marker='v', markersize=8, markerfacecolor='white',
        markeredgecolor='#555555', linestyle='none')], ['Output-cap lower bound'],
        loc='upper center', bbox_to_anchor=(.53, .94), frameon=False, fontsize=19)
    basename = args.output.name
    for suffix in ('png', 'pdf'):
        fig.savefig(args.output / (basename + '.' + suffix), dpi=300,
            bbox_inches='tight', pad_inches=.06, facecolor='white')
    plt.close(fig)
    fingerprint = hashlib.sha256(source).hexdigest()
    lines = ['# 有效前瞻距离的观察分布', '',
        '左图为 Qwen3.6-35B-A3B BF16 + Native MTP；右图为 Qwen3-235B-A22B Q4_K_M GGUF + EAGLE3。每模型四配置，各 GSM8K 前 10 条、固定 256 输出，一次；offload、eager、无预取、ready hits、4 loops、显存比例 0.9、默认 46 个物理核。', '',
        'D 是一次启动后，直到旧前瞻完全失效前，最终被 CPU 接受的旧候选 token 数；部分重查的有效旧前缀继续累计。每次启动等权，包含初始启动和重启；不是每个请求等权，也不把相关重启当成独立抽样。', '',
        'D 为非负整数。主图仅对 D>0 的观察值做高斯核平滑，固定带宽 8 token，所有方法使用同一带宽；采用边界反射，不把密度漏到负值。密度除以全部尝试数，主曲线面积为 1-P(D=0)。D=0 的离散概率放在小图中，不能把它画成有限高度的概率密度尖峰。', '',
        '小图 4R/4C/6R/6C 分别对应 top-4 重放、top-4 重查、top-6 重放、top-6 重查，与曲线颜色一致。', '',
        '曲线是实际观察到的 D 下界的描述性平滑分布，不是去除右截断影响后的潜在真实 D 概率密度。空心倒三角标记因 256 输出上限而结束观察的下界；这些值参与“观察值分布”，但不是在该处失败，尤其长距离处的峰不能解释为失败集中。真实未结束距离可以更大。', '',
        '没有置信区间或独立性假设。样本量小且重启相关；平滑只帮助比较，细小峰形不作统计推断。本实验不包含性能测量。', '',
        f'数据源：[{args.metrics.name}]({args.metrics.resolve()})', '',
        f'数据 SHA-256：`{fingerprint}`', '',
        '| 模型 / 配置 | 尝试数 | D=0 | 输出上限截断 |', '| --- | ---: | ---: | ---: |']
    for row in receipt:
        lines.append(f"| {row['case']} | {row['attempts']} | {row['zero_probability']:.2%} | {row['output_censored']} |")
    continuation = ' ' + chr(92)
    lines += ['', '复现命令：', '', '```bash',
        'TMPDIR=' + str(args.metrics.resolve().parents[1] / 'tmp') + continuation,
        'MPLCONFIGDIR=' + str(args.metrics.resolve().parents[1] / '.figure_work/matplotlib') + continuation,
        '/home/fanya/FreeToken-upstream-integration-20261023/.venv/bin/python' + continuation,
        '  ' + str(Path(__file__).resolve()) + continuation,
        '  --metrics ' + str(args.metrics.resolve()) + continuation,
        '  --output ' + str(args.output.resolve()) + continuation,
        '  --validation ' + str(args.validation.resolve()) + continuation,
        '  --bandwidth ' + str(args.bandwidth), '```']
    (args.output / (basename + '.md')).write_text('\n'.join(lines) + '\n')
    assert args.metrics.read_bytes() == source
    args.validation.write_text(json.dumps(dict(valid=True, source_sha256=fingerprint,
        source_unchanged=True, bandwidth_tokens=args.bandwidth, reflected_boundary=True,
        zero_mass_separate=True, scope='descriptive observed lower-bound distribution; no censor correction',
        cases=receipt), indent=2) + '\n')
    print(json.dumps(dict(output=str(args.output.resolve()), source_sha256=fingerprint)))


if __name__ == '__main__':
    main()
