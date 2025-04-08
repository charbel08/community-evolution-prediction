import pandas as pd
import networkx as nx
import torch

from src.cluster import cluster
from src.trajectories import get_trajectories
from src.features import get_rnn_data_padded
from src.train import train

def run_experiment(config):

    # Read graph and dataframe
    df = pd.read_parquet(config["data_path"])
    G = nx.read_gexf(config["graph_path"])

    node_to_community = cluster(G, method=config["clustering"])
    trajectories, graphs_per_snapshot = get_trajectories(node_to_community, G, df, config["clustering"])

    if config["features"]["load_save"]:
        sequences, labels = torch.load(config["features"]["X_path"]), torch.load(config["features"]["t_path"])
    else:
        sequences, labels = get_rnn_data_padded(trajectories, graphs_per_snapshot, save=config["features"])
    
    train(sequences, labels)

    
