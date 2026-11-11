"""Budgeted KL calibration for prefill-only and original combined MoDES."""

import argparse
import copy
import json
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parent
SNAPSHOT = ROOT / 'runtime_snapshot'
sys.path[:0] = [str(SNAPSHOT / 'python'), str(SNAPSHOT / 'benchmarks')]
from run_experiment import ALIASES, MODEL, calibration_rows, now, save


def main(name, alpha, deadline, scopes=('prefill', 'prefill_decode')):
    started = time.time()
    scopes = tuple(scopes)
    if not scopes or len(set(scopes)) != len(scopes) or any(
            scope not in ('prefill', 'prefill_decode') for scope in scopes):
        raise ValueError(f'invalid calibration scopes: {scopes}')
    if name not in ALIASES:
        raise ValueError(f'inactive calibration dataset: {name}')
    root = ROOT / 'calibration' / name
    root.mkdir(parents=True, exist_ok=True)
    call_id = f'{"_".join(scopes)}_{time.time_ns()}'
    call_path = root / 'calls' / f'{call_id}.json'
    save(call_path, dict(complete=False, call_id=call_id, requested_scopes=scopes,
         started_utc=now(), started_unix=started, deadline=deadline))

    import torch
    from transformers import AutoProcessor, Qwen3VLMoeForConditionalGeneration
    from modes_calibration import CalibrationModel, make_inputs, normalize_importance, tau_grids
    sys.path.insert(0, str(ROOT / 'calibration_support'))
    from modes_search import coarse_to_fine, length_stratified_subset, validate_full_target
    from freetoken.moe.modes import MODES_COMMIT, validate_calibration
    from freetoken.moe.tablemoe_artifacts import finish_stage

    processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True)
    records = []
    for row in calibration_rows(name):
        row = copy.deepcopy(row)
        reference = ROOT / 'references' / name / (row['id'].replace(':', '_') + '.txt')
        reference.parent.mkdir(parents=True, exist_ok=True)
        if name != 'mmw':
            answer = str(row['metadata']['answer']).strip()
            if not answer or answer in ('None', 'nan'):
                raise ValueError(f'missing ground-truth reference: {row["id"]}')
            reference.write_text(answer)
        if not reference.is_file() or not reference.read_text().strip():
            raise ValueError(f'new full-route reference is missing: {row["id"]}')
        row['metadata']['reference_answer'] = str(reference)
        inputs, mask = make_inputs(processor, row,
                  dict(size=dict(shortest_edge=602112, longest_edge=4014080)), 32768)
        path = root / 'inputs' / (row['id'].replace(':', '_') + '.pt')
        path.parent.mkdir(exist_ok=True)
        torch.save(dict(inputs=inputs, mask=mask), path)
        records.append(dict(id=row['id'], path=str(path), tokens=inputs['input_ids'].numel(),
                            answer_tokens=int(mask.sum()), category=row['metadata'].get('category')))
    save(root / 'inputs.json', dict(records=records, content_hashes_computed=False,
         reference_kind='new full-route generation' if name == 'mmw' else 'ground truth'))
    if time.time() >= deadline:
        raise TimeoutError('preparation exhausted the calibration budget')
    model = Qwen3VLMoeForConditionalGeneration.from_pretrained(MODEL, dtype=torch.float16,
                  attn_implementation='sdpa', local_files_only=True).to('cuda').eval()
    binding = json.loads((ROOT / 'checkpoint_binding.json').read_text())

    def load(record):
        value = torch.load(record['path'], map_location='cpu', weights_only=True)
        return {key: tensor.to('cuda') for key, tensor in value['inputs'].items()}, value['mask'].to('cuda')

    for scope_index, scope in enumerate(scopes):
        scope_started = time.time()
        scope_deadline = scope_started + (deadline - scope_started) / (len(scopes) - scope_index)
        folder = root / scope
        folder.mkdir(exist_ok=True)
        adapter = CalibrationModel(model, processor.tokenizer, logits_chunk=32, sequence_chunk=256, scope=scope)
        layers = sorted(adapter.layers)
        losses = {layer: dict(text=0., visual=0.) for layer in layers}
        teachers = {}
        selected = []
        desired = len(records)
        importance_budget = .65 * (scope_deadline - scope_started)
        for index, record in enumerate(records):
            if index >= desired:
                break
            if time.time() >= scope_started + importance_budget:
                raise TimeoutError('importance calculation exhausted its reserved budget')
            inputs, mask = load(record)
            teacher = adapter.hidden(inputs, mask).cpu()
            teachers[record['id']] = teacher
            for layer in layers:
                for modality in ('text', 'visual'):
                    if time.time() >= scope_started + importance_budget:
                        raise TimeoutError('importance calculation exhausted its reserved budget')
                    candidate = adapter.hidden(inputs, mask, ablation=(layer, modality))
                    losses[layer][modality] += adapter.kl(teacher, candidate)
                    save(folder / 'progress.json', dict(stage='importance', id=record['id'],
                         completed_samples=index, requested_samples=desired, layer=layer, modality=modality,
                         elapsed_seconds=time.time() - scope_started, updated_utc=now()))
            selected.append(record)
            if index == 1:
                per_sample = (time.time() - scope_started) / 2
                minimum = 6 if name in ('mmw', 'mmmu') else 4
                desired = min(len(records), max(minimum, int(importance_budget / per_sample)))
            print('IMPORTANCE', name, scope, len(selected), desired, flush=True)
            del inputs, mask
        importance = normalize_importance(losses, sum(record['answer_tokens'] for record in selected))
        save(folder / 'importance.json', dict(importance=importance, raw_losses=losses,
             selected_ids=[record['id'] for record in selected], elapsed_seconds=time.time() - scope_started))
        subset_size = min(len(selected), 4 if name == 'rwqa' else 6)
        subset = length_stratified_subset(selected, subset_size)
        measurements = {}

        def evaluate(text, visual, records_to_use):
            key = (text, visual, tuple(record['id'] for record in records_to_use))
            if key in measurements:
                return measurements[key]
            loss = total = skipped = tokens = 0
            phases = {stage: dict(total=0, skipped=0) for stage in ('prefill', 'decode')}
            for record in records_to_use:
                if time.time() >= scope_deadline:
                    raise TimeoutError('threshold evaluation reached the calibration deadline')
                inputs, mask = load(record)
                adapter.total_routes = adapter.skipped_routes = 0
                adapter.stage_routes = {stage: dict(total=0, skipped=0) for stage in phases}
                candidate = adapter.hidden(inputs, mask, tau=dict(text=text, visual=visual), importance=importance)
                loss += adapter.kl(teachers[record['id']], candidate)
                total += adapter.total_routes
                skipped += adapter.skipped_routes
                tokens += record['answer_tokens']
                for stage in phases:
                    for field in phases[stage]:
                        phases[stage][field] += adapter.stage_routes[stage][field]
                del inputs, mask, candidate
            if not total:
                raise ValueError('calibration has no eligible route denominator')
            measurements[key] = (loss / tokens, skipped / total)
            save(folder / 'last_measurement.json', dict(tau=dict(text=text, visual=visual),
                 ids=[record['id'] for record in records_to_use], loss=loss / tokens,
                 skip_fraction=skipped / total, phases=phases, updated_utc=now()))
            return measurements[key]

        grids = tau_grids(100)
        search_until = scope_started + .85 * (scope_deadline - scope_started)
        candidate = coarse_to_fine(*grids, lambda text, visual: evaluate(text, visual, subset), 1 - alpha,
                                   coarse_points=7, fine_points=5,
                                   continue_phase=lambda phase: time.time() < search_until)
        result = validate_full_target(candidate['tau'], dict(text=grids[0][-1], visual=grids[1][-1]),
                  lambda text, visual: evaluate(text, visual, selected), 1 - alpha,
                  max_evaluations=8, tolerance=.01,
                  continue_evaluation=lambda: time.time() < scope_deadline)
        save(folder / 'search.json', dict(result, scope=scope, target=1 - alpha,
             coarse_to_fine=candidate, original_grid_search_complete=False,
             selected_ids=[record['id'] for record in selected], heldout_samples_used=0))
        if not result['target_tolerance_met']:
            raise RuntimeError(f'{name} {scope}: target tolerance not met')
        config = dict(schema='freetoken_modes_v1', checkpoint=binding,
                      checkpoint_id='Qwen/Qwen3-VL-30B-A3B-Instruct',
                      checkpoint_association='metadata matched preserved offline-asset provenance',
                      modes_source=dict(repository='https://github.com/ModelTC/MoDES', commit=MODES_COMMIT),
                      control=False, calibration_info=dict(dataset=ALIASES[name], scope=scope,
                      target_skip=1 - alpha, actual_skip=result['skip_proportion'], samples=len(selected),
                      runtime_snapshot=str(SNAPSHOT), heldout_samples_used=0,
                      original_grid_search_complete=False),
                      layers={str(layer): dict(importance=importance[layer], tau=result['tau']) for layer in layers})
        validate_calibration(config, layers)
        output = folder / 'modes_config'
        save(output / 'modes_config.json', config)
        finish_stage(output, dict(scope=scope, selected_ids=[record['id'] for record in selected],
                     target=1 - alpha, runtime_snapshot=str(SNAPSHOT)), [output / 'modes_config.json'])
        save(folder / 'complete.json', dict(complete=True, scope=scope, tau=result['tau'],
             actual_skip=result['skip_proportion'], elapsed_seconds=time.time() - scope_started,
             target_skip=1 - alpha, call_id=call_id, deadline=scope_deadline,
             finished_utc=now(), shared_across_stages=scope == 'prefill_decode'))
        teachers.clear()
    finished = time.time()
    save(call_path, dict(complete=True, call_id=call_id, requested_scopes=scopes,
         started_unix=started, finished_utc=now(), elapsed_seconds=finished - started,
         deadline=deadline, completed_before_deadline=finished <= deadline))
    completed = []
    for scope in ('prefill', 'prefill_decode'):
        path = root / scope / 'complete.json'
        if path.is_file():
            value = json.loads(path.read_text())
            if value.get('complete') and value.get('target_skip') == 1 - alpha:
                completed.append(scope)
    calls = [json.loads(path.read_text()) for path in (root / 'calls').glob('*.json')]
    finished_calls = [call for call in calls if call.get('complete')]
    save(root / 'complete.json', dict(complete=len(completed) == 2,
         completed_scopes=completed, last_requested_scopes=scopes, last_call_id=call_id,
         last_call_elapsed_seconds=finished - started,
         elapsed_seconds=sum(call['elapsed_seconds'] for call in finished_calls),
         finished_utc=now(), deadline=deadline,
         completed_before_deadline=all(call['completed_before_deadline'] for call in finished_calls)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=tuple(ALIASES), required=True)
    parser.add_argument('--alpha', type=float, required=True)
    parser.add_argument('--deadline', type=float, required=True)
    parser.add_argument('--scope', choices=['both', 'prefill', 'prefill_decode'], default='both')
    arguments = parser.parse_args()
    scopes = ('prefill', 'prefill_decode') if arguments.scope == 'both' else (arguments.scope,)
    main(arguments.dataset, arguments.alpha, arguments.deadline, scopes=scopes)
