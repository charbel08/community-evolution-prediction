import pandas as pd
import networkx as nx

from src.cluster import cluster
from src.trajectories import get_trajectories

def run_experiment(config):

    # Read graph and dataframe
    df = pd.read_parquet(config["data_path"])
    G = nx.read_gexf(config["graph_path"])

    node_to_community = cluster(G, method=config["clustering"])
    trajectories = get_trajectories(node_to_community, G, df, config["clustering"])

    

    
