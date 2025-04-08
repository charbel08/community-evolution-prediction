import pandas as pd
from pathlib import Path
import networkx as nx

RAW_DATA_PATH = Path("data/raw/twitter/interactions_march_23-24_2025.csv")
PROCESSED_DATA_PATH = Path("data/processed/interactions_cleaned.parquet")

def load_df(filepath: Path) -> pd.DataFrame:
    df = pd.read_csv(
        filepath, 
        sep="\t", 
        quotechar='"', 
        encoding="utf-8", 
        dtype={"from_user_id": str, "to_user_id": str}
    )
    df = df[df['to_user_id'].notna() & df['from_user_id'].notna()]
    return df

def get_network_graph(df: pd.DataFrame) -> nx.Graph:
    # Create graph
    G = nx.from_pandas_edgelist(
        df,
        source='from_user_id',
        target='to_user_id',
        edge_attr='created_at',
        create_using=nx.Graph()
    )

    # Simplify graph
    G.remove_edges_from(nx.selfloop_edges(G))
    G.remove_nodes_from(list(nx.isolates(G)))

    return G

def get_snapshot_df(df: pd.DataFrame) -> pd.DataFrame:
    df['datetime'] = pd.to_datetime(df['created_at'])
    df['snapshot'] = df['datetime'].dt.to_period("h")
    sorted_snapshots = sorted(df['snapshot'].unique())
    snapshot_id_map = {snap: idx for idx, snap in enumerate(sorted_snapshots)}
    df['snapshot_id'] = df['snapshot'].map(snapshot_id_map)
    return df


def main():
    print(f"Generating networkx graph from {RAW_DATA_PATH}...")

    # Load DataFrame and Graph
    df = load_df(RAW_DATA_PATH)
    G = get_network_graph(df)
    print(f"Graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

    # Get temporal snapshots
    df = get_snapshot_df(df)

    # Save graph
    graph_path = Path("data/processed/twitter/graphs/twitter_graph.gexf")
    nx.write_gexf(G, graph_path)
    print(f"Graph saved to {graph_path}")

    # Save processed DataFrame
    df_path = Path("data/processed/twitter/snapshots/twitter_snapshots.parquet")
    df.to_parquet(df_path, index=False)
    print(f"DataFrame with snapshots saved to {df_path}")

if __name__ == "__main__":
    main()
