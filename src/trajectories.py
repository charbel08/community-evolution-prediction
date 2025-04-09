import pandas as pd
import networkx as nx
import numpy as np

from typing import Dict
from collections import defaultdict


def get_graphs_per_snapshot(G: nx.Graph, df: pd.DataFrame, cluster_method: str, dataset: str) -> Dict:

    graphs_by_snapshot = {}
    for snapshot, group in df.groupby('snapshot_id'):

        if dataset == "twitter":
            source, destination = 'from_user_id', 'to_user_id'
        elif dataset == "reddit":
            source, destination = 'SOURCE_SUBREDDIT', 'TARGET_SUBREDDIT'
        
        group = group[[source, destination]]

        G_snapshot = nx.from_pandas_edgelist(
            group,
            source=source,
            target=destination,
            create_using=nx.Graph()
        )

        # Simplify snapshot graph
        G_snapshot.remove_edges_from(nx.selfloop_edges(G_snapshot))
        G_snapshot.remove_nodes_from(list(nx.isolates(G_snapshot)))

        for node in G_snapshot.nodes:
            if node in G.nodes and cluster_method in G.nodes[node]:
                G_snapshot.nodes[node][cluster_method] = G.nodes[node][cluster_method]

        # filename = f"/content/gdrive/My Drive/COMP511/KDD/graphs/private_X/kmeans/snapshot_graphs/{snapshot}.gexf"
        # nx.write_gexf(G_snapshot, filename)

        graphs_by_snapshot[snapshot] = G_snapshot
    
    return graphs_by_snapshot


def get_community_graphs_per_snapshot(node_to_community, graphs_by_snapshot):

    communities = defaultdict(set)
    for node, comm_id in node_to_community.items():
        communities[comm_id].add(node)

    community_graphs_by_snapshot = {}

    for snapshot, G_snapshot in graphs_by_snapshot.items():
        community_subgraphs = {}

        for comm_id, community_nodes in communities.items():
            # Intersect global community nodes with nodes present in the snapshot
            nodes_in_snapshot = community_nodes & G_snapshot.nodes
            if nodes_in_snapshot:
                subgraph = G_snapshot.subgraph(nodes_in_snapshot).copy()
                community_subgraphs[comm_id] = subgraph

        community_graphs_by_snapshot[snapshot] = community_subgraphs
    
    return community_graphs_by_snapshot


def get_trajectories(node_to_community: Dict[str, int], G: nx.Graph, df: pd.DataFrame, cluster_method: str, dataset: str):

    graphs_per_snapshot = get_graphs_per_snapshot(G, df, cluster_method, dataset)
    community_graphs_per_snapshot = get_community_graphs_per_snapshot(node_to_community, graphs_per_snapshot)

    trajectories = defaultdict(list)
    for snapshot in range(len(graphs_per_snapshot) - 1):
        community = community_graphs_per_snapshot[snapshot]
        community_next = community_graphs_per_snapshot[snapshot+1]
        for comm_id, community_graph in community.items():
            if (community_graph, snapshot) not in trajectories[comm_id]:
                trajectories[comm_id].append((community_graph, snapshot))
            if comm_id in community_next:
                trajectories[comm_id].append((community_next[comm_id], snapshot+1))
    
    # trajectory_lengths = [len(items) for items in trajectories.values()]
    # print(np.max(trajectory_lengths))
    # print(np.min(trajectory_lengths))

    return trajectories, graphs_per_snapshot

