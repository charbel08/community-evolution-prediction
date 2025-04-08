import networkx as nx
import numpy as np
import kmapper as km
import torch

from tqdm import tqdm
from sklearn.manifold import SpectralEmbedding, TSNE

def extract_node_features(G):
    # Compute features for each node
    degree = dict(G.degree())
    clustering = nx.clustering(G)
    closeness = nx.closeness_centrality(G)
    betweenness = nx.betweenness_centrality(G)

    node_ids = []
    features = []

    for node in G.nodes():
        node_ids.append(node)
        # Create a feature vector for this node
        # You can adjust which features to include
        feat = [
            degree[node],
            clustering[node],
            closeness[node],
            betweenness[node],
            # eigenvector[node]  # Uncomment if using eigenvector centrality
        ]
        features.append(feat)

    X = np.array(features)
    return X, node_ids

def extract_graph_features(G):
    n = G.number_of_nodes()
    e = G.number_of_edges()
    density = nx.density(G)

    # try:
    #     diameter = nx.diameter(G)
    # except (nx.NetworkXError, nx.exception.NetworkXNoPath):
    #     diameter = 0

    try:
        avg_sp = nx.average_shortest_path_length(G)
    except (nx.NetworkXError, nx.exception.NetworkXNoPath):
        avg_sp = 0

    # Degree Centrality
    dc = list(nx.degree_centrality(G).values())
    max_dc = max(dc) if dc else 0
    min_dc = min(dc) if dc else 0

    # Closeness Centrality
    cc = list(nx.closeness_centrality(G).values())
    max_cc = max(cc) if cc else 0
    min_cc = min(cc) if cc else 0

    # Betweenness Centrality
    # bc = list(nx.betweenness_centrality(G).values())
    # max_bc = max(bc) if bc else 0
    # min_bc = min(bc) if bc else 0

    return np.array([
        n, e, density, avg_sp,
        max_dc, min_dc, max_cc, min_cc,
        # max_bc, min_bc
    ])


def get_snapshot_mapper_features(graphs_per_snapshot):
    
    mapper_features = {}
    for snapshot, G_snapshot in tqdm(graphs_per_snapshot.items(), desc='Generating Mapper Snapshots'):
        if G_snapshot.number_of_nodes() < 5:
            mapper_features[snapshot] = np.zeros((8,))
        else:
            X, _ = extract_node_features(G_snapshot)
            # A = nx.to_numpy_array(G_snapshot)
            # embedding = SpectralEmbedding(n_components=2)  # You can tune components
            # X = embedding.fit_transform(A)

            mapper = km.KeplerMapper()
            # print(features.shape)
            lens = mapper.fit_transform(X, projection=TSNE())
            mapper_graph = km.to_networkx(
                mapper.map(lens, X, cover=km.Cover(n_cubes=2, perc_overlap=0.4))
            )

            mapper_features[snapshot] = extract_graph_features(mapper_graph)
            
    return mapper_features


def get_rnn_data_padded(trajectories, graphs_per_snapshot, metric="edges", save=None):
    sequences = []
    labels = []

    mapper_features = get_snapshot_mapper_features(graphs_per_snapshot)

    total_snapshots = max(
        max(snapshot for _, snapshot in items)
        for items in trajectories.values()
    ) + 1

    feature_dim = len(extract_graph_features(next(iter(trajectories.values()))[0][0])) * 2

    total = 0
    for value in trajectories.values():
        total += len(value)

    with tqdm(total=total, desc="Generating Community Features") as pbar:
      for community, items in trajectories.items():
          sorted_items = sorted(items, key=lambda tup: tup[1])

          feature_seq = np.zeros((total_snapshots, feature_dim), dtype=np.float32)
          label_seq = np.full((total_snapshots,), fill_value=-1, dtype=np.float32)  # BCE expects float

          snapshot_to_metric = {}

          for graph, snapshot in sorted_items:
              features = extract_graph_features(graph)

            #   x, node_ids = extract_node_features(graph)
            #   mapper = km.KeplerMapper()
            #   lens = mapper.fit_transform(x, projection=TSNE())
            #   mapper_graph = km.to_networkx(
            #       mapper.map(lens, x, cover=km.Cover(n_cubes=2, perc_overlap=0.3))
            #   )

            #   if mapper_graph.number_of_nodes() == 0:
            #       features_mapped = np.zeros_like(features)
            #   else:
            #       features_mapped = extract_graph_features(mapper_graph)

              combined_features = np.concatenate((features, mapper_features[snapshot]))
              feature_seq[snapshot] = combined_features
            #   mask_seq[snapshot] = True

              current_metric = graph.number_of_edges() if metric == "edges" else graph.number_of_nodes()
              snapshot_to_metric[snapshot] = current_metric
              pbar.update(1)

          # Assign labels: compare t vs t+1
          for t in sorted(snapshot_to_metric.keys()):
              if t + 1 in snapshot_to_metric:
                  metric_t = snapshot_to_metric[t]
                  metric_tp1 = snapshot_to_metric[t + 1]
                  if metric_tp1 > metric_t:
                      label_seq[t] = 1  # growth
                  elif metric_tp1 < metric_t:
                      label_seq[t] = 0  # shrinkage

          sequences.append(torch.tensor(feature_seq, dtype=torch.float32))
          labels.append(torch.tensor(label_seq, dtype=torch.float32))
    
    sequences, labels = torch.stack(sequences), torch.stack(labels)

    valid_row_mask = (labels != -1).any(dim=1)  # shape: (B,), True if at least one valid label

    # Apply the mask to filter out rows with only -1
    labels = labels[valid_row_mask]
    sequences = sequences[valid_row_mask]

    if save:
        torch.save(labels, save['t_path'])
        torch.save(sequences, save['X_path'])

    return sequences, labels

