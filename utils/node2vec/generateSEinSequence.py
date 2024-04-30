import os
import node2vec
import numpy as np
import networkx as nx
from gensim.models import Word2Vec

# Parameters
is_directed = True
p = 2
q = 1
num_walks = 100
walk_length = 80
dimensions = 64
iter = 1000

def read_graph(edgelist):
    G = nx.read_edgelist(
        edgelist, nodetype=int, data=(('weight',float),),
        create_using=nx.DiGraph())
    return G

def learn_embeddings(walks, dimensions, output_file):
    walks = [list(map(str, walk)) for walk in walks]
    model = Word2Vec(
        walks, vector_size=dimensions, window=10, min_count=0, sg=1,
        workers=8, epochs=iter)
    model.wv.save_word2vec_format(output_file)

def process_adj_directories(adj_directories):
    for directory in adj_directories:
        for file in os.listdir(directory):
            if file.endswith("_edge_list.txt"):
                adj_file = os.path.join(directory, file)
                output_file = os.path.join(directory, file.replace("_edge_list.txt", "_SE.txt"))
                generate_embeddings(adj_file, output_file)

def generate_embeddings(adj_file, output_file):
    nx_G = read_graph(adj_file)
    G = node2vec.Graph(nx_G, is_directed, p, q)
    G.preprocess_transition_probs()
    walks = G.simulate_walks(num_walks, walk_length)
    learn_embeddings(walks, dimensions, output_file)
    print(f"Embeddings generated and saved as {output_file}")

# Example usage:
adj_directories = ['E:/xie/Sensor Files/5. dataset/215 WB', 
                   'E:/xie/Sensor Files/5. dataset/215 EB',
                  ]
process_adj_directories(adj_directories)
