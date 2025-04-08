from typing import Dict

import community.community_louvain as community_louvain
import networkx as nx

def louvain(G: nx.Graph) -> Dict:
    return community_louvain.best_partition(G)


# Add community labels to graph
# nx.set_node_attributes(G, node_to_community, name='louvain')

def cluster(G, method="louvain"):
    print(f"Clustering with {method}")
    if method == "louvain":
        node_to_community = louvain(G)
    else:
        raise ValueError(f"Unsupported clustering method: {method}")
    
    nx.set_node_attributes(G, node_to_community, name=method)
    return node_to_community
