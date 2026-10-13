#!/usr/bin/env python3
from pathlib import Path
import hashlib,csv
P=Path(__file__).resolve().parent
root=Path.cwd()
paths=sorted((P.parent.parent/'grid').glob('observations_qs_*.csv'))
paths+=sorted((root/'current_development/market_model/results/QS').glob('rates_sco_*.csv'))
paths += [P.parent.parent/'grid/transition_errors.csv',P/'extract_snapshot.tar.gz']
with open(P/'input_manifest.csv','w',newline='') as f:
 w=csv.writer(f);w.writerow(['path','bytes','sha256'])
 for p in paths:
  w.writerow([str(p.relative_to(root)),p.stat().st_size,hashlib.sha256(p.read_bytes()).hexdigest()])
print(f'Hashed {len(paths)} immutable analysis inputs.')
