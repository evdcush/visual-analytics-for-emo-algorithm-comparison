import json, sys, time
import numpy as np
from sklearn.decomposition import PCA

import torch
from geomloss import SamplesLoss


sinkhorn = SamplesLoss('sinkhorn', blur=0.05, p=1)
assert len(sys.argv) == 2
print(f'Convert data of problem {sys.argv[1]}.')


def get_distance(dataA: dict, dataB: dict, same: bool):
  distAB = { k: {} for k in dataA.keys() }
  distBA = { k: {} for k in dataB.keys() }
  DATA_A_VEC = { k: torch.tensor(v) for k, v in dataA.items() }
  DATA_B_VEC = { k: torch.tensor(v) for k, v in dataB.items() }
  if same:
    record = []

  for kA, vA in DATA_A_VEC.items():
    for kB, vB in DATA_B_VEC.items():
      if same and (kA, kB) in record:
        distAB[kA][kB] = distAB[kB][kA]
        distBA[kB][kA] = distBA[kA][kB]
        continue
      res = sinkhorn(vA, vB)
      distAB[kA][kB] = res.item()
      distBA[kB][kA] = res.item()
      if same:
        record.append((kB, kA))

  return distAB, distBA


PROBLEM = sys.argv[1]
CONFIG = None

with open(f'./{PROBLEM}/index.json') as fc:
  CONFIG = dict(json.load(fc))

assert CONFIG is not None

if CONFIG['pca'] is True:
  with open(f'./{PROBLEM}/{CONFIG["origin"]}/{CONFIG["reference"]}') as fi:
    PF = np.array(json.load(fi))
    pca = PCA(n_components=2)
    res = pca.fit_transform(PF)
  with open(f'./{PROBLEM}/{CONFIG["display"]}/{CONFIG["reference"]}', 'w') as fo:
    json.dump(res.tolist(), fo, indent=2)

OBJ_DATA = {}

for name, f in dict(CONFIG['algorithms']).items():
  with open(f'./{PROBLEM}/{CONFIG["origin"]}/{f}') as fi:
    src = dict(json.load(fi))
    OBJ_DATA[name] = src['result']['obj']
    if CONFIG['pca'] is True:
      print(f'Calculating PCA of {name}...', end='')
      st = time.monotonic()
      for k, v in src['result']['obj'].items():
        src['result']['obj'][k] = (pca.transform(np.array(v))).tolist()
  if CONFIG['pca'] is True:
    with open(f'./{PROBLEM}/{CONFIG["display"]}/{f}', 'w') as fo:
      json.dump(src, fo, indent=2)
    ed = time.monotonic()
    print(f'Done, {(ed - st):.3f}s used.')

calc_dist = input('Enter "Y" if you want calculate the sinkhorn distance matrix: ')
if calc_dist != 'Y':
  exit(0)

for iA, kA in enumerate(OBJ_DATA.keys()):
  for iB, kB in enumerate(OBJ_DATA.keys()):
    if iB < iA:
      continue
    print(f'Calculating sinkhorn distance of {kA} and {kB}...', end='')
    st = time.monotonic()
    distAB, distBA = get_distance(OBJ_DATA[kA], OBJ_DATA[kB], kA == kB)
    with open(f'./{PROBLEM}/{CONFIG["distance"]}/{kA}_{kB}_{PROBLEM}_distance.json', 'w') as fo:
      json.dump(distAB, fo, indent=2)
    if kA != kB:
      with open(f'./{PROBLEM}/{CONFIG["distance"]}/{kB}_{kA}_{PROBLEM}_distance.json', 'w') as fo:
        json.dump(distBA, fo, indent=2)
    ed = time.monotonic()
    print(f'Done, {(ed - st):.3f}s used.')