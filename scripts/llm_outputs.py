import networkx as nx
import matplotlib.pyplot as plt

# def make_llm_call
# Model 1: Via CHAT GPT LLM
from openai import OpenAI
from pprint import pprint
import re
import pickle
import time


def get_gpt_cluster_label_predictions(prompt):

    #read from text file if it exists
    # if os.path.exists('gpt_output.txt'):
    #     with open('gpt_output.txt', 'r') as f:
    #         sense_keys = f.read()
    #     return sense_keys.split('\n')

    # Query GPT to get the word sense keys
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "Infer and perform accurately the task of cluster labeling given top cluster node's features"},
            {"role": "user", "content": prompt}
        ],
        model="gpt-3.5-turbo-16k",  # Use the appropriate model, gpt-4-32k not supported, or "gpt-3.5-turbo-16k"
        max_tokens=1500,  # Adjust the max tokens as needed, was a 1000
        temperature=0.0  # Set temperature to 0 for deterministic results
    )

    # Parse the response to get the sense keys
    # sense_keys = response.choices[0].message.content

    #Parse the response to get the cluster labels
    cluster_labels = response.choices[0].message.content

    #write to text file
    # with open('gpt_output.txt', 'w') as f:
    #     f.write(cluster_labels)

    print("Cluster labels are: ", cluster_labels)
    # sense_keys = sense_keys.split('\n')
    return cluster_labels

def get_top_nodes_features_by_cluster(G, num_nodes=5, cluster_key='ward'): #clusterkey could be 'louvain' or 'kmeans' or 'ward' or 'HDBSCAN'
    top_nodes_features = {}
    # Get unique cluster IDs
    cluster_ids = set(nx.get_node_attributes(G, cluster_key).values())
    # cluster_ids = set(nx.get_node_attributes(G, 'louvain').values())
    print("Total number of clusters: ", len(cluster_ids))
    #set timeout for 3 seconds
    # Pause execution for 3 seconds
    if(len(cluster_ids) > 100):
        num_nodes = 2
    time.sleep(3)
    print("Execution resumed after 3 seconds.")
    

    for cluster_id in cluster_ids:
        # Get nodes belonging to the current cluster
        cluster_nodes = [node for node, data in G.nodes(data=True) if data.get(cluster_key) == cluster_id]
        # cluster_nodes = [node for node, data in G.nodes(data=True) if data.get('louvain') == cluster_id]
        # Sort nodes by degree within the cluster
        top_nodes = sorted(cluster_nodes, key=lambda x: G.degree(x), reverse=True)[:num_nodes]
        top_nodes_features[cluster_id] = top_nodes
    return top_nodes_features

def get_prompt(top_nodes_per_cluster, G):
  prompt = ""
#   nodePerCluster = 5
#   edgesPerNode = 2
#   if(len(top_nodes_per_cluster) > 100):
#     nodePerCluster = 8
#     edgesPerNode = 3
#   token_size = get_token_size_per_edge_feature(len(top_nodes_per_cluster), nodesPerCluster=nodePerCluster, edgesPerNode=edgesPerNode)
  for cluster_id, top_nodes in top_nodes_per_cluster.items():
    prompt += f"Cluster {cluster_id}: "
    for node in top_nodes:
      print("Node features is: ", G.nodes[node])
      # print("Edge feature is:", G.edges[node])
      # print("Top 2 edge features based on weight of edge: ", sorted(G.edges(node, data=True), key=lambda x: x[2]['weight'], reverse=True)[:2])
      top_2_edges_features = sorted(G.edges(node, data=True), key=lambda x: x[2]['weight'], reverse=True)[:2]
      print("Top 2 edge features based on weight of edge: ", top_2_edges_features)

      # Get outgoing edges for the node, sort them, and select top 2
      # edges = sorted(G.edges(node, data=True), key=lambda x: x[2]['weight'], reverse=True)
      # top_edges = edges[:2]  # Select top 2 edges or less if there are fewer

      #just use the weight and comment feature within the 3rd element of the tuple
      top_2_edges_features = [(u, v, data['weight'], data['comment']) for u, v, data in top_2_edges_features]
      #and trim the comment text to max length of 20
    #   print("Token size is: ", token_size)
    #   time.sleep(5)
      top_2_edges_features = [(u, v, weight, comment[:20]) for u, v, weight, comment in top_2_edges_features]

      # print("Top 2 edge features based on weight of edge: ", top_edges)


      # prompt += f"{G.nodes[node]}, "
      prompt += f"Edges features for a node: {top_2_edges_features}, "

    prompt += f"For Cluster: {cluster_id} done\n"

  #say instructions ie give this cluster of users/nodes a label based on its text based node features (ie post content and so on)
  prompt += "Give this cluster of edges belonging to nodes: a label based on its text based node features (ie post content and so on)"
  prompt += " and your response should be in the following format: json of keys as cluster ids and value as your generated cluster labels"
  return prompt

def get_token_size_per_edge_feature(clusterSize, nodesPerCluster=5, edgesPerNode=2):
    #ideally for 45 clusters, with lets say a max of 5 nodes per cluster, and 2 edges per node, we can allow 20 characters for comments edge feature
    scaling_constant = 15000

    # Calculate the token size
    total_edges = clusterSize * nodesPerCluster * edgesPerNode
    token_size = scaling_constant // total_edges
    
    # Ensure token size is at least 1 and at most 2048
    token_size = max(1, min(token_size, 2048))
    return token_size

from huggingface_hub import login, whoami

import torch
import networkx as nx
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

def get_graph_embedding_from_llama(
    graph: nx.Graph,
    model_name: str = "meta-llama/Llama-2-7b-hf",
    layer_idx: int = -2,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> torch.Tensor:
    """
    Converts a NetworkX graph into a serialized text prompt,
    feeds it to LLaMA, and returns a graph embedding from a hidden layer.
    
    Args:
        graph (nx.Graph): The input graph (can include 'type', 'relation', 'weight' attributes).
        model_name (str): Name of the LLaMA model to use.
        layer_idx (int): Index of the hidden layer to extract embeddings from.
        device (str): Device to run the model on.

    Returns:
        torch.Tensor: Graph embedding of shape (1, hidden_dim).
    """
    token = "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" #full read access token
    login(token=token)
    print(whoami())  # Should return your Hugging Face account details
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True, use_auth_token=True).to(device)
    model.eval()

    # Step 1: Serialize graph
    graph_text = "Graph:\n"

    # Node descriptions
    for node_id, attrs in graph.nodes(data=True):
        node_type = attrs.get("type", "entity")
        graph_text += f"Node {node_id} is a {node_type}.\n"

    # Edge descriptions
    for u, v, attrs in graph.edges(data=True):
        relation = attrs.get("relation", None)
        weight = attrs.get("weight", None)

        # Build relation string
        if relation:
            relation_text = f"{relation}"
        else:
            relation_text = "is connected to"

        # Add weight info if available
        if weight is not None:
            graph_text += f"Node {u} {relation_text} Node {v} with weight {weight}.\n"
        else:
            graph_text += f"Node {u} {relation_text} Node {v}.\n"

    # Step 2: Tokenize and run model
    inputs = tokenizer(graph_text, return_tensors="pt", truncation=True, max_length=2048).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        hidden_states = outputs.hidden_states

    # Step 3: Extract and pool embedding
    hidden_layer = hidden_states[layer_idx]
    graph_embedding = hidden_layer.mean(dim=1)  # Shape: (1, hidden_dim)

    return graph_embedding



def save_result_toFile(result, filename):
    """
    Save the result to a text file.
    
    Args:
        result (str): The result to save.
        filename (str): The name of the file to save the result in.
    """
    with open(filename, 'w') as f:
        f.write(result)
    print(f"Result saved to {filename}")




# implement main
if __name__ == "__main__":

    print("Hello")

    # #get name of graph file as input to this py file
    graph_name = input("Enter the name of the graph file (without .gpickle): ")
    #set default if no input
    if not graph_name:
        graph_name = "G_tiktok_clustered"

    cluster_key = input("Enter the cluster key (default is 'ward'): ")
    #set default if no input
    if not cluster_key:
        cluster_key = "ward"

    # *** Preprocessing and prompt generation LLM
    #load G_tiktok_clustered and full raw from file
    # G_tiktok_cluster = nx.read_gpickle("G_tiktok_clustered.gpickle")
    # G_tiktok_w = nx.read_gpickle("G_tiktok.gpickle")

    # load clustered graph object from file
    G = pickle.load(open(graph_name+'.pickle', 'rb'))

    # top_nodes_per_cluster_tiktok = get_top_nodes_features_by_cluster(G_tiktok_cluster)
    # top_nodes_per_cluster_pvt_dataset = get_top_nodes_features_by_cluster(G_pvt_cluster)
    top_nodes_per_cluster = get_top_nodes_features_by_cluster(G, cluster_key=cluster_key)
    print(top_nodes_per_cluster)
    # print(top_nodes_per_cluster_pvt_dataset)

    #ensure there is text and weight in edges
    edge_example = list(G.edges(data=True))[6] # can use
    print("\nEdge with comment text as weight:")
    print(edge_example)

    prompt= get_prompt(top_nodes_per_cluster, G)
    # prompt_pvt_dataset = get_prompt(top_nodes_per_cluster_pvt_dataset, G_pvt_cluster)
    #print prompts and their token size
    print(prompt)
    # print(prompt_pvt_dataset)
    print(len(prompt))
    # print(len(prompt_pvt_dataset))



    # *** Call the LLM to get cluster labels
    # Load the model
    # OpenAI API key

    #API please add your own if possible to test the code
    #this key will be deactivated after assignment is graded
    # api_key = "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"
    api_key = "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"
    # openai.api_key
    # Initialize the OpenAI client
    client = OpenAI(
        api_key=api_key  # This is the default and can be omitted
    )
    cluster_labels_tiktok = get_gpt_cluster_label_predictions(prompt)
    # cluster_labels_pvt_dataset = get_gpt_cluster_label_predictions(prompt_pvt_dataset)

    #sample prints
    print(cluster_labels_tiktok)
    # print(cluster_labels_pvt_dataset)
    #save the result to a file
    save_result_toFile(cluster_labels_tiktok, "cluster_labels_" + graph_name + "_" + cluster_key+ ".txt")


    # *** Graph LLM Embedding
    # Create a simple example graph
    # G = nx.Graph()
    # G.add_node("A", type="person")
    # G.add_node("B", type="person")
    # G.add_node("C", type="organization")
    # G.add_node("D", type="organization")

    # G.add_edge("A", "B", relation="knows")
    # G.add_edge("B", "C", relation="works_for")
    # G.add_edge("C", "D", relation="partner_of")




    # # Get graph embedding
    # #sample model name that doesn't need user info agreeement
    # # model_name = "openlm-research/open_llama_7b"
    # embedding = get_graph_embedding_from_llama(G)
    # print("Graph Embedding Shape:", embedding.shape)
    # print("Graph Embedding: ", embedding)

    # # Now for our actual graph
    # # Get graph embedding
    # # Save the embedding to a file
    # torch.save(embedding, "graph_embedding_" + graph_name + "_" + cluster_key + ".pt")



