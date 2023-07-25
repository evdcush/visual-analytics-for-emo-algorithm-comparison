import networkx as nx
import numpy as np
import torch
import hdbscan
from torch_cluster import knn_graph


DataMatrix = list[list[dict]]


def dict_to_matrix(data: dict) -> np.ndarray:
  len_data = len(data)
  res = np.zeros((len_data, len_data))
  for i, k in enumerate(data.values()):
    for j, v in enumerate(k.values()):
      res[i, j] = v
  return res


def get_cluster(matrix: torch.Tensor) -> list[int]:
  clusterer = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=1)
  clusterer.fit(matrix)
  return clusterer.labels_.tolist()


def calc_graph(data: DataMatrix, name: list[str]) -> dict:
  mat_data = [[dict_to_matrix(j) for j in i] for i in data]
  mat_data = np.concatenate(tuple([np.concatenate(tuple(i), axis=1) for i in mat_data]), axis=0)
  mat = torch.tensor(mat_data)

  nodes, edges = [], []

  for i in range(len(data)):
    for it in data[i][0].keys():
      nodes.append({ 'name': name[i], 'frame': it })

  old_edges = []
  for k in range(1, 11):
    new_edges = knn_graph(mat, k=k)
    for i in range(new_edges.shape[1]):
      u, v = new_edges[0, i].item(), new_edges[1, i].item()
      if (u, v) not in old_edges:
        edges.append([u, v, k])
        old_edges.append((u, v))

  graph = nx.Graph()
  graph.add_nodes_from(np.arange(mat.shape[0]).tolist())
  graph.add_edges_from(old_edges)
  pos = nx.kamada_kawai_layout(graph)
  for i, p in enumerate(pos.values()):
    nodes[i]['pos'] = p.tolist()

  clusters = get_cluster(mat)

  return { 'nodes': nodes, 'edges': edges, 'clusters': clusters }