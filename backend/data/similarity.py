import json
import sys
import numpy as np
from openTSNE import TSNE

import torch
from dtw import *
from geomloss import SamplesLoss

sinkhorn = SamplesLoss('sinkhorn', blur=0.05, p=1)
assert len(sys.argv) == 2
print(f'Compute runs similarity for data of problem {sys.argv[1]}.')
PROBLEM = sys.argv[1]
CONFIG = None

with open(f'./{PROBLEM}/index.json') as fc:
    CONFIG = dict(json.load(fc))

assert CONFIG is not None

origin = {}
for name, f in dict(CONFIG['algorithms']).items():
    with open(f'./{PROBLEM}/{CONFIG["origin"]}/{f}') as fi:
        origin[name] = dict(json.load(fi))

distance = {}
# index keys of origin to number
distance['runs'] = list(origin.keys())
distance['metrics'] = {}
distance['projections'] = {}
# generate seven zero numpy arrays in a list
zeros = [np.zeros((len(origin.keys()), len(origin.keys()))) for _ in range(6)]
IGD_euclidean, HV_euclidean, IGD_best_frame_sinkhorn, HV_best_frame_sinkhorn, IGD_dtw, HV_dtw = zeros
metric_list = ['IGD_euclidean', 'HV_euclidean', 'IGD_best_frame_sinkhorn', 'HV_best_frame_sinkhorn', 'IGD_dtw', 'HV_dtw']
for i in range(len(origin.keys())):
    for j in range(i + 1, len(origin.keys())):
        run_i = list(origin.keys())[i]
        run_j = list(origin.keys())[j]
        IGD_i = np.array(list(origin[run_i]['metric']['IGD'].values()))
        IGD_j = np.array(list(origin[run_j]['metric']['IGD'].values()))
        HV_i = np.array(list(origin[run_i]['metric']['HV'].values()))
        HV_j = np.array(list(origin[run_j]['metric']['HV'].values()))
        resulti = list(origin[run_i]['result']['obj'].values())
        resultj = list(origin[run_j]['result']['obj'].values())
        IGD_euclidean[i][j] = np.linalg.norm(IGD_i - IGD_j)
        IGD_euclidean[j][i] = IGD_euclidean[i][j]
        HV_euclidean[i][j] = np.linalg.norm(HV_i - HV_j)
        HV_euclidean[j][i] = HV_euclidean[i][j]
        IGD_dtw[i][j] = dtw(IGD_i, IGD_j).distance
        IGD_dtw[j][i] = IGD_dtw[i][j]
        HV_dtw[i][j] = dtw(HV_i, HV_j).distance
        HV_dtw[j][i] = HV_dtw[i][j]
        # frames_sinkhorn_all = []
        # for k in range(len(resulti)):
        #     frames_sinkhorn_all.append(sinkhorn(
        #         torch.tensor(resulti[k]), torch.tensor(resultj[k])))
        # frames_sinkhorn[i][j] = np.mean(frames_sinkhorn_all)
        # frames_sinkhorn[j][i] = frames_sinkhorn[i][j]
        IGD_best_frame_sinkhorn[i][j] = sinkhorn(
            torch.tensor(resulti[np.argmin(IGD_i)]),
            torch.tensor(resultj[np.argmin(IGD_j)]))
        IGD_best_frame_sinkhorn[j][i] = IGD_best_frame_sinkhorn[i][j]
        HV_best_frame_sinkhorn[i][j] = sinkhorn(
            torch.tensor(resulti[np.argmax(HV_i)]),
            torch.tensor(resultj[np.argmax(HV_j)]))
        HV_best_frame_sinkhorn[j][i] = HV_best_frame_sinkhorn[i][j]
for metric in metric_list:
    distance['metrics'][metric] = locals()[metric].tolist()
for metric in metric_list:
    distance['projections'][metric] = TSNE().fit(locals()[metric]).tolist()


with open(f'./{PROBLEM}/runs_similarity.json', 'w') as fo:
    json.dump(distance, fo)
