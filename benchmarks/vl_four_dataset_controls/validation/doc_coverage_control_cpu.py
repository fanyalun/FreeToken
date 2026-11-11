import importlib
import json
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
run_cell = importlib.import_module('run_cell')
manager = importlib.import_module('manager')
fixture = ROOT / 'validation' / f'doc_coverage_controls_{time.time_ns()}'
fixture.mkdir()
gate = str(fixture / 'coverage_ready.json')
override = str(fixture / 'repaired_assets')
(fixture / 'table_assets.json').write_text(json.dumps({'MMLongBench_DOC': override}))
(fixture / 'scheduling_policy.json').write_text(json.dumps({'additional_dependencies': {
    'generate_MMLongBench_DOC_table': [gate],
    'generate_MMLongBench_DOC_tablemoe': [gate],
}}))
run_cell.ROOT = fixture
manager.ROOT = fixture
assert run_cell.asset_root('MMLongBench_DOC') == Path(override)
assert str(run_cell.asset_root('BabyVision')).endswith('/BabyVision/assets')
assert 'mmmu_offline_bs16_seed42_split60' in str(run_cell.asset_root('MMMU_DEV_VAL'))
jobs = manager.jobs()
guarded = {job['id'] for job in jobs if gate in job['dependencies']}
assert guarded == {'generate_MMLongBench_DOC_table', 'generate_MMLongBench_DOC_tablemoe'}
assert not manager.complete(gate)
Path(gate).write_text(json.dumps({'complete': False}))
assert not manager.complete(gate)
Path(gate).write_text(json.dumps({'complete': True}))
assert manager.complete(gate)
command, _ = run_cell.command('MMLongBench_DOC', 'table', 0)
assert command[command.index('--tablemoe-cache-dir') + 1] == str(Path(override) / 'cache')
(fixture / 'scheduling_policy.json').write_text('{}')
assert all(gate not in job['dependencies'] for job in manager.jobs())
print(json.dumps({'complete': True, 'checks': 9, 'gpu_used': False, 'fixture': str(fixture)}))
