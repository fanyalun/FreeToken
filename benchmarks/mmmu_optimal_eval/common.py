"""Frozen MMMU protocol helpers shared by preparation, workers, and scoring."""

import ast
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import sys


DATASET = 'MMMU_DEV_VAL'
CELLS = ('mmmu_tablemoe', 'mmmu_modes', 'mmmu_schedule')


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + '.partial')
    partial.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    partial.replace(path)


def write_rows(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(value, ensure_ascii=False) + '\n' for value in values))


def source_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def answer_aliases(value):
    if isinstance(value, str):
        value = value.strip()
        if value.startswith('['):
            try:
                value = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                pass
    values = value if isinstance(value, (list, tuple)) else [value]
    aliases = list(dict.fromkeys(str(item).strip() for item in values if item is not None))
    if not aliases or any(not item or item.lower() == 'nan' for item in aliases):
        raise ValueError('reference answer is empty')
    return aliases


def answer_only(row):
    row = dict(row)
    choice = row['metadata']['question_type'] == 'multiple-choice'
    options = [key for key in 'ABCDEFGHI' if isinstance(row['metadata'].get(key), str)
               and row['metadata'][key].strip()]
    instruction = ('Output only the single uppercase letter of the correct option (' + ', '.join(options)
                   + '). Do not output reasoning, analysis, an explanation, the option text, or any other text.'
                   if choice else 'Output only the final short answer. Do not output reasoning, analysis, '
                   'an explanation, or any other text.')
    message = [dict(part) for part in row['vlmeval_message']]
    last = max(i for i, part in enumerate(message) if part['type'] == 'text')
    message[last]['value'] += '\n\n' + instruction
    row.update(base_vlmeval_message=row['vlmeval_message'], vlmeval_message=message,
               prompt_protocol_id='mmmu_answer_only_ab_v1', answer_only_instruction=instruction,
               answer_format='single_option_letter' if choice else 'short_final_answer', valid_options=options)
    return row


def select_calibration(calibration, evaluation, per_category=2, seed=42):
    evaluation_ids = {row['id'] for row in evaluation}
    evaluation_groups = {row['group_id'] for row in evaluation}
    evaluation_images = {image['path'] for row in evaluation for image in row['images']}
    if any(row['id'] in evaluation_ids or row['group_id'] in evaluation_groups or
           any(image['path'] in evaluation_images for image in row['images']) for row in calibration):
        raise ValueError('calibration and evaluation overlap')
    groups = defaultdict(list)
    for row in calibration:
        groups[row['group_id']].append(row)
    categories = sorted({row['metadata']['category'] for row in calibration})
    rng = random.Random(seed)
    queues = {}
    for category in categories:
        keys = sorted(key for key, group in groups.items()
                      if any(row['metadata']['category'] == category for row in group))
        rng.shuffle(keys)
        queues[category] = keys
    selected, used, counts = [], set(), Counter()
    for _ in range(per_category):
        for category in categories:
            if counts[category] >= per_category:
                continue
            key = next((key for key in queues[category] if key not in used), None)
            if key is None:
                raise ValueError('not enough isolated calibration groups')
            used.add(key)
            for row in sorted(groups[key], key=lambda row: row['id']):
                selected.append(row)
                counts[row['metadata']['category']] += 1
    return selected, dict(seed=seed, per_category_quota=per_category, selected_ids=[r['id'] for r in selected],
                          selected_groups=sorted(used), category_counts=dict(counts),
                          quota_deviation={key: counts[key] - per_category for key in categories},
                          heldout_samples_used=0, shared_groups_kept_together=True)


def job(cell):
    if cell not in CELLS:
        raise ValueError('unknown MMMU configuration')
    return dict(id=cell, dataset='mmmu', method='modes' if cell == 'mmmu_modes' else 'tablemoe',
                alpha=.8, h=4, slots=3072, skip_lookup=cell == 'mmmu_schedule')


def load_runner(root, task):
    protocol = read(root / 'protocol.json')
    source = root / 'controller_source.py'
    if source_hash(source) != protocol['source_hashes']['controller_source.py']:
        raise ValueError('frozen controller source changed')
    spec = importlib.util.spec_from_file_location('mmmu_frozen_controller', source)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    runner.ROOT = root
    runner.SNAPSHOT = Path(protocol['runtime_snapshot'])
    runner.JUDGE_SNAPSHOT = Path(protocol['judge_runtime_snapshot'])
    runner.ALIASES = {'mmmu': DATASET}
    runner.LIMITS = {'mmmu': 256}
    runner.persist = lambda: save(root / 'tasks' / (task + '_runtime.json'), runner.STATE)
    sys.path[:0] = [str(runner.SNAPSHOT / 'python'), str(runner.SNAPSHOT / 'benchmarks')]
    return runner


def predictions(root, cell):
    expected = rows(root / 'prepare' / DATASET / 'evaluation.jsonl')
    actual = rows(root / 'cells' / cell / 'requests.jsonl')
    if len(expected) != 420 or len({r['id'] for r in expected}) != 420:
        raise ValueError('frozen evaluation coverage changed')
    if [r['id'] for r in actual] != [r['id'] for r in expected]:
        raise ValueError('prediction coverage or order differs')
    if any(r['max_tokens'] != 256 or not r['natural_eos'] or
           r['usage']['completion_tokens'] > 256 for r in actual):
        raise ValueError('generation protocol differs')
    return expected, actual


def aggregate(expected, records):
    indices = {str(row['index']) for row in expected}
    if len(indices) != len(expected) or len(records) != len(expected) or \
            {str(row['index']) for row in records} != indices:
        raise ValueError('score coverage lost or duplicated indices')
    if any(type(row['hit']) is not int or row['hit'] not in (0, 1) for row in records):
        raise ValueError('invalid per-sample score')
    correct = sum(row['hit'] for row in records)
    return dict(correct=correct, scoring_denominator=len(expected), accuracy_pct=100 * correct / len(expected))


def parse_judgment(text):
    text = text.strip()
    if text.startswith('```') and text.endswith('```'):
        text = text.split('\n', 1)[1].rsplit('```', 1)[0].strip()
    value = json.loads(text)
    if not isinstance(value, dict) or set(value) != {'correct'} or type(value['correct']) is not bool:
        raise ValueError('judge must return one boolean correct field')
    return int(value['correct'])
