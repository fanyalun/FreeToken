import json
import sys
from pathlib import Path

import calibrate_modes as calibration


before=sys.path.copy()
module=calibration.load_calibrator('MMMU_DEV_VAL')
assert sys.path==before
assert module.ROOT==calibration.ROOT
assert module.SNAPSHOT==calibration.SNAPSHOT
assert module.ALIASES=={'MMMU_DEV_VAL':'MMMU_DEV_VAL'}
assert 'run_experiment' not in sys.modules
assert module.calibration_rows is calibration.select
rows=calibration.select('MMMU_DEV_VAL')
assert len(rows)==12
assert len({row['category'] for row in rows})==6
path=calibration.ROOT/'validation/modes_import_cpu.json'
calibration.write_json(path,dict(passed=True,imports_restore_sys_path=True,
    old_run_experiment_not_imported=True,selected_mmmu_samples=len(rows),categories=6))
print(path)
