import json
import logging
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
src_path = repo_root / "src"
if str(src_path) not in sys.path:
    sys.path.append(str(src_path))

from energy_forecaster.utils.logging_config import configure_logging

configure_logging()
logger = logging.getLogger(__name__)

path = Path('recorder_list.out')
if not path.exists():
    raise SystemExit('recorder_list.out not found')

with path.open() as f:
    data = json.load(f)

entries = data.get('result', [])
ids = [entry.get('statistic_id') for entry in entries if entry.get('statistic_id')]
logger.info("total statistics: %s", len(ids))
needles = [
    'sensor.house_consumption',
    'sensor.pv_power',
    'sensor.goodwe_house_consumption',
    'sensor.goodwe_pv_power',
]
for needle in needles:
    logger.info("%s present: %s", needle, needle in ids)
logger.info('sample containing "goodwe":')
for sid in ids:
    if sid and 'goodwe' in sid.lower():
        logger.info('  %s', sid)
        break