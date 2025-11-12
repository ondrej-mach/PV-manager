import json
from pathlib import Path

path = Path('recorder_list.out')
if not path.exists():
    raise SystemExit('recorder_list.out not found')

with path.open() as f:
    data = json.load(f)

entries = data.get('result', [])
ids = [entry.get('statistic_id') for entry in entries if entry.get('statistic_id')]
print(f"total statistics: {len(ids)}")
needles = [
    'sensor.house_consumption',
    'sensor.pv_power',
    'sensor.goodwe_house_consumption',
    'sensor.goodwe_pv_power',
]
for needle in needles:
    print(f"{needle}:", needle in ids)
print('sample containing "goodwe":')
for sid in ids:
    if sid and 'goodwe' in sid.lower():
        print('  ', sid)
        break