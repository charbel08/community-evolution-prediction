import torch
import pandas as pd
import networkx as nx
import numpy as np

from src.cluster import cluster
from src.trajectories import get_trajectories
from src.features import get_rnn_data_padded
from src.train import train
from src.plot import plot_training_summary
from src.utils import set_seed

def run_experiments(config):

    # Read graph and dataframe
    df = pd.read_parquet(config["data_path"])
    G = nx.read_gexf(config["graph_path"])
    
    train_stats, test_accs = [], []
    for run in range(1, config["num_runs"] + 1):
        # set_seed(run)
        if run >= 2:
            config["features"]["load_save"] = True
        node_to_community = cluster(G, method=config["clustering"])
        trajectories, graphs_per_snapshot = get_trajectories(node_to_community, G, df, config["clustering"], config["dataset"])

        if config["features"]["load_save"]:
            sequences, labels = torch.load(config["features"]["X_path"]), torch.load(config["features"]["t_path"])
        else:
            sequences, labels = get_rnn_data_padded(trajectories, graphs_per_snapshot, save=config["features"])
            
        train_losses, train_accs, val_accs, test_acc = train(sequences, labels, run, config)
        test_accs.append(test_acc)
        train_stats.append(np.stack([train_losses, train_accs, val_accs]))

    # Train stats
    train_stats = np.stack(train_stats)
    train_means, train_stds = np.mean(train_stats, axis=0), np.std(train_stats, axis=0)
    plot_training_summary(train_means, train_stds, config["experiment_name"])

    # Test stats
    test_accs = np.array(test_accs)
    test_mean = np.mean(test_accs)
    test_std = np.std(test_accs)
    print(f"Test Accuracy: {test_mean:.4f} ± {test_std:.4f}")
    