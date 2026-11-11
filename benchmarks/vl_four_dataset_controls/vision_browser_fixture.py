"""Run an unscored CPU deployment and native three-viewport screenshot fixture."""

import asyncio
import importlib.util
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

from PIL import Image


ROOT = Path(__file__).resolve().parent
RUNTIME = Path('/data/fanya/tablemoe/vl_four_dataset_bs1_20261111/vision2web_runs_reasoning_off_20261111/smoke/fused/official_runtime')


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


async def exercise(folder):
    source = ROOT / 'runtime_snapshot/benchmarks/bench_vision2web.py'
    spec = importlib.util.spec_from_file_location('fixture_adapter', source)
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    sys.path.insert(0, str(RUNTIME))
    from vision2web.evaluation.engine import EvaluationEngine
    from vision2web.evaluation.functional_tester import FunctionalTester
    workspace = folder / 'workspace_source'
    workspace.mkdir()
    (workspace / 'index.html').write_text('''<!doctype html><html><head><meta name="viewport" content="width=device-width,initial-scale=1"><style>
body{margin:0;background:#f0f3f8;color:#16304f;font:24px sans-serif;min-height:1100px}
header{background:#16304f;color:white;padding:40px}main{padding:36px;max-width:900px}
.tile{background:#f6c65b;padding:32px;margin:20px 0}footer{margin-top:500px;background:#318a82;color:white;padding:24px}
</style></head><body><header>VISION2WEB CPU FIXTURE</header><main><h1>Deployment and screenshot check</h1>
<div class="tile">Unscored infrastructure fixture</div><p>Three native viewports; no model request.</p></main>
<footer>Persistent output fixture</footer></body></html>''')
    (workspace / 'start.sh').write_text('#!/bin/sh\nexec python3 -m http.server 3000 --bind 0.0.0.0\n')
    workflows = json.loads(Path('/data/fanya/datasets/vision2web_level1/webpage/aihw/workflow.json').read_text())
    save(folder / 'workflow.json', workflows)
    screenshots = {}

    class CaptureOnly:
        async def score_all_prototypes(self, **kwargs):
            screenshots.update({name: len(data) for name, data in kwargs['actual_pages'].items()})

    engine = EvaluationEngine.__new__(EvaluationEngine)
    engine.logger = logging.getLogger('vision_browser_fixture')
    engine.functional_tester = FunctionalTester(model='unused-cpu-fixture', api_key='EMPTY',
                                                base_url='http://127.0.0.1:1')
    engine.visual_scorer = CaptureOnly()
    sandbox = adapter.persistent_sandbox(dict(output=str(folder),
        sandbox_image='vision2web-sandbox:freetoken-d80a4a3'), 'browser_fixture', engine.logger)
    container = None
    try:
        container = await sandbox.create_container(workspace)
        if not container or not await sandbox.start_container(workspace):
            raise RuntimeError('fixture container failed to start')
        inspection = json.loads(subprocess.check_output(['docker', 'inspect', container], text=True))[0]
        save(folder / 'container.json', dict(id=container, image=inspection['Config']['Image'],
            device_requests=inspection['HostConfig'].get('DeviceRequests'), mounts=inspection['Mounts'],
            temporary_environment=[s for s in inspection['Config']['Env'] if s.startswith(('TMPDIR=', 'TMP=', 'TEMP='))]))
        await asyncio.wait_for(engine._run_evaluation(container, workspace,
            SimpleNamespace(path=workspace), workflows, dict(name='unscored_cpu_fixture')), timeout=180)
    finally:
        await sandbox.cleanup_all()
        if container:
            result = subprocess.run(['docker', 'inspect', container], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            save(folder / 'cleanup.json', dict(own_container=container, removed=result.returncode != 0))
    if set(screenshots) != {'desktop', 'tablet', 'mobile'}:
        raise RuntimeError(f'native screenshot coverage incomplete: {screenshots}')
    checks = []
    for workflow in workflows:
        name = workflow['summary']
        path = workspace / 'test_results/prototypes' / f'{name}_actual.png'
        with Image.open(path) as image:
            image.load()
            dimensions = image.size
            extrema = image.convert('RGB').getextrema()
            valid = (image.format == 'PNG' and dimensions[0] == workflow['resolution']['width']
                     and dimensions[1] >= workflow['resolution']['height']
                     and any(low < high for low, high in extrema))
        checks.append(dict(viewport=name, path=str(path), size=path.stat().st_size,
                           dimensions=dimensions, valid=valid))
    if not all(item['valid'] for item in checks):
        raise RuntimeError(f'invalid screenshots: {checks}')
    return checks


def main():
    folder = ROOT / 'validation' / f'vision_browser_fixture_{time.time_ns()}'
    (folder / 'tmp').mkdir(parents=True)
    for name in ('TMPDIR', 'TMP', 'TEMP'):
        os.environ[name] = str(folder / 'tmp')
    logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(folder / 'run.log'), logging.StreamHandler()])
    receipt = dict(complete=False, fixture_only=True, benchmark_score=False, gpu_started=False,
                   model_called=False, official_runtime=str(RUNTIME), evidence_root=str(folder))
    try:
        receipt['screenshots'] = asyncio.run(exercise(folder))
        receipt['complete'] = True
    except BaseException as error:
        receipt['error'] = repr(error)
    save(folder / 'complete.json', receipt)
    save(ROOT / 'validation/vision_browser_fixture.json', receipt)
    print(json.dumps(receipt, indent=2))
    raise SystemExit(0 if receipt['complete'] else 1)


if __name__ == '__main__':
    main()
