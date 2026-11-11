"""Preserve MMMU membership and add the requested direct-answer instruction."""

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys


ROOT = Path(os.environ.get('MMMU_AB_ROOT', Path(__file__).resolve().parent))
SOURCE = Path('/home/fanya/FreeToken_TableMoE/artifacts/tablemoe_sensitivity_preserve_vlmeval_20261110')
SNAPSHOT = SOURCE / 'runtime_snapshot'
KIT = Path('/home/fanya/tablemoe/third_party/VLMEvalKit')
sys.path[:0] = [str(SNAPSHOT / 'python'), str(SNAPSHOT / 'benchmarks'), str(KIT)]


def main():
    from tablemoe_data import api_messages, read_rows, write_rows
    from tablemoe_vlmeval import kit_fingerprint, load_dataset, load_prompt_model, select_prompt

    fingerprint = kit_fingerprint(KIT)
    expected = json.loads((SOURCE / 'prepare/kit.json').read_text())
    if fingerprint != expected:
        raise ValueError('VLMEvalKit changed since the frozen evaluation preparation')
    original = read_rows(SOURCE / 'prepare/MMMU_DEV_VAL/evaluation.jsonl')
    if len(original) != 420:
        raise ValueError('expected the original complete 420-sample evaluation')
    dataset = load_dataset(KIT, 'MMMU_DEV_VAL', True)
    model = load_prompt_model(KIT, dataset, 'qwen3_vl')
    prepared = []
    sources, types = Counter(), Counter()
    for old in original:
        frame = dataset.data[dataset.data['index'].astype(str) == str(old['index'])]
        if len(frame) != 1:
            raise ValueError('native MMMU index is missing or duplicated')
        base, prompt_source = select_prompt(dataset, frame.iloc[0], model)
        if base != old['vlmeval_message']:
            raise ValueError('the native prompt differs from the preserved base prompt')
        if not all(Path(part['value']).is_file() for part in base if part['type'] == 'image'):
            raise ValueError('a preserved MMMU image is missing')
        choice = old['metadata']['question_type'] == 'multiple-choice'
        if choice:
            allowed = [letter for letter in 'ABCDEFGHI'
                       if isinstance(old['metadata'].get(letter), str) and old['metadata'][letter].strip()]
            instruction = ('Output only the single uppercase letter of the correct option (' +
                           ', '.join(allowed) + '). Do not output reasoning, analysis, an explanation, '
                           'the option text, or any other text.')
        else:
            allowed = []
            instruction = ('Output only the final short answer. Do not output reasoning, analysis, '
                           'an explanation, or any other text.')
        message = [dict(part) for part in base]
        text_index = max(index for index, part in enumerate(message) if part['type'] == 'text')
        message[text_index]['value'] += '\n\n' + instruction
        row = dict(old, base_vlmeval_message=base, vlmeval_message=message, messages=api_messages(message),
                   prompt_source=prompt_source + ' plus user-requested direct-answer instruction',
                   prompt_protocol_id='mmmu_answer_only_ab_v1', answer_format='single_option_letter' if choice else 'short_final_answer',
                   valid_options=allowed, answer_only_instruction=instruction)
        prepared.append(row)
        sources[prompt_source] += 1
        types[old['metadata']['question_type']] += 1
    if [row['id'] for row in prepared] != [row['id'] for row in original]:
        raise ValueError('the preserved evaluation membership or order changed')
    destination = ROOT / 'prepare/MMMU_DEV_VAL'
    destination.mkdir(parents=True, exist_ok=True)
    write_rows(destination / 'evaluation.jsonl', prepared)
    protocol = dict(dataset='MMMU_DEV_VAL', complete=True, samples=420, question_types=dict(types),
                    base_prompt_sources=dict(sources), user_authorized_answer_only=True,
                    base_prompt_protocol=str(SOURCE / 'prepare/MMMU_DEV_VAL/prompt_protocol.json'),
                    source_files={str(path.relative_to(KIT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                  for path in [KIT / 'vlmeval/vlm/qwen3_vl/prompt.py', KIT / 'vlmeval/dataset/image_mcq.py']},
                    split_membership_and_order_preserved=True, image_order_preserved=True,
                    source_snapshot=str(SNAPSHOT), max_tokens=256,
                    offline_table_prompt_unchanged=True, image_content_hashes_computed=False,
                    offline_tensor_content_hashes_computed=False,
                    prepared_utc=datetime.now(timezone.utc).isoformat())
    (destination / 'prompt_protocol.json').write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + '\n')
    (ROOT / 'prepare/kit.json').write_text(json.dumps(fingerprint, indent=2) + '\n')
    (ROOT / 'prepare/complete.json').write_text(json.dumps(dict(complete=True, samples=420, question_types=dict(types)), indent=2) + '\n')
    print(json.dumps(dict(complete=True, samples=420, question_types=dict(types), base_prompt_sources=dict(sources))))


if __name__ == '__main__':
    main()
