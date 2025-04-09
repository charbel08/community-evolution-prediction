import pandas as pd
from pathlib import Path
import networkx as nx
import argparse
from pathlib import Path

def load_df(filepath: str) -> pd.DataFrame:
    _, _, dataset, _ = filepath.split("/")

    if dataset == "twitter":
        df = pd.read_csv(
            filepath, 
            sep="\t", 
            quotechar='"', 
            encoding="utf-8", 
            dtype={"from_user_id": str, "to_user_id": str}
        )
        df = df[df['to_user_id'].notna() & df['from_user_id'].notna()]
    elif dataset == "reddit":
        df = pd.read_csv(filepath, sep='\t')
    return df

def get_network_graph(df: pd.DataFrame, dataset: str) -> nx.Graph:


    if dataset == "twitter":
        G = nx.from_pandas_edgelist(
            df,
            source='from_user_id',
            target='to_user_id',
            edge_attr='created_at',
            create_using=nx.Graph()
        )
    elif dataset == "reddit":
        G = nx.from_pandas_edgelist(
            df, 
            source='SOURCE_SUBREDDIT', 
            target='TARGET_SUBREDDIT', 
            edge_attr=True, 
            create_using=nx.Graph()
        )

    # Simplify graph
    G.remove_edges_from(nx.selfloop_edges(G))
    G.remove_nodes_from(list(nx.isolates(G)))

    return G

def get_snapshot_df(df: pd.DataFrame, dataset: str, granularity: str) -> pd.DataFrame:

    if dataset == "twitter":
        df['datetime'] = pd.to_datetime(df['created_at'])
    elif dataset == "reddit":
        df['datetime'] = pd.to_datetime(df['TIMESTAMP'], format='%Y-%m-%d %H:%M:%S')
        
    df['snapshot'] = df['datetime'].dt.to_period(granularity)
    sorted_snapshots = sorted(df['snapshot'].unique())
    snapshot_id_map = {snap: idx for idx, snap in enumerate(sorted_snapshots)}
    df['snapshot_id'] = df['snapshot'].map(snapshot_id_map)

    return df


def main(raw_data_file: str, dataset: str, granularity: str):
    print(f"Generating networkx graph from {raw_data_file}...")

    # Load DataFrame and Graph
    df = load_df(f"data/raw/{dataset}/{raw_data_file}")
    G = get_network_graph(df, dataset)
    print(f"Graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

    # Get temporal snapshots
    df = get_snapshot_df(df, dataset, granularity)

    # Save graph
    graph_path = Path(f"data/processed/{dataset}/graphs/{dataset}_graph_{granularity}.gexf")
    nx.write_gexf(G, graph_path)
    print(f"Graph saved to {graph_path}")

    # Save processed DataFrame
    df_path = Path(f"data/processed/{dataset}/snapshots/{dataset}_snapshots_{granularity}.parquet")
    df.to_parquet(df_path, index=False)
    print(f"DataFrame with snapshots saved to {df_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data_file", required=True, type=str, help="Path to the raw data.")
    parser.add_argument("--dataset", required=True, type=str, help="Name of the dataset")
    parser.add_argument("--granularity", required=True, type=str, help="Granularity of the snapshots")

    args = parser.parse_args()
    main(args.raw_data_file, args.dataset, args.granularity)
