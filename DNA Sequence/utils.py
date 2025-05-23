import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from GraphTsetlinMachine.graphs import Graphs

def is_valid_sequence(seq):
    valid_bases = {'A', 'C', 'G', 'T'}
    return all(base in valid_bases for base in seq)

def filter_valid_sequences(data):
    return data[data['sequence'].apply(is_valid_sequence)]

def get_sequence_length_stats(filtered_data):
    max_lengths = filtered_data.groupby('label')['sequence'].apply(lambda seqs: seqs.apply(len).max())
    min_lengths = filtered_data.groupby('label')['sequence'].apply(lambda seqs: seqs.apply(len).min())
    return max_lengths, min_lengths

def get_unique_bases_per_label(filtered_data):
    return filtered_data.groupby('label')['sequence'].apply(lambda x: set(''.join(x)))

def sample_balanced_data(filtered_data, labels, n=1799):
    df_filtered = filtered_data[filtered_data['label'].isin(labels)]
    return df_filtered.groupby('label').apply(lambda x: x.sample(n=n, random_state=42)).reset_index(drop=True)

def create_graph_data(df, args, label_mapping, is_train=True, init_graph=None):
    graphs = Graphs(
        len(df),
        symbols=['GGA', 'ACA', 'AAA', 'TAT', 'GCC', 'AAC', 'GTG', 'CCC', 'TCG', 'TCT', 'GAA', 'GAC', 'GGG', 'TTG', 'GAT', 
                 'TGA', 'GGT', 'TAG', 'TGC', 'GGC', 'CGC', 'CGA', 'TTT', 'CTT', 'GAG', 'CCA', 'TAA', 'AGC', 'GCA', 'CCT', 
                 'ATT', 'TAC', 'CGT', 'CTG', 'GTC', 'AAG', 'AGA', 'TTA', 'GCT', 'CAA', 'GTA', 'CAT', 'ACC', 'AAT', 'CTC', 
                 'GTT', 'CAC', 'ATC', 'TCA', 'TGG', 'TCC', 'AGT', 'CAG', 'CCG', 'ACT', 'ATG', 'TGT', 'TTC', 'GCG', 'AGG', 
                 'ACG', 'CTA', 'ATA', 'CGG'],
        hypervector_size=args.hypervector_size,
        hypervector_bits=args.hypervector_bits,
        double_hashing=args.double_hashing
    ) if is_train else Graphs(len(df), init_with=init_graph)

    for graph_id in range(len(df)):
        sequence = df.iloc[graph_id]['sequence']
        less_seq = min(len(sequence), args.max_sequence_length)
        num_nodes = max(0, less_seq - 2)
        graphs.set_number_of_graph_nodes(graph_id, num_nodes)
    graphs.prepare_node_configuration()

    for graph_id in range(len(df)):
        for node_id in range(graphs.number_of_graph_nodes[graph_id]):
            edges = 2 if 0 < node_id < graphs.number_of_graph_nodes[graph_id] - 1 else 1
            graphs.add_graph_node(graph_id, node_id, edges)
    graphs.prepare_edge_configuration()

    labels = np.empty(len(df), dtype=np.uint32)

    for graph_id in range(len(df)):
        label = df.iloc[graph_id]['label']
        labels[graph_id] = label_mapping.get(label, -1)
        sequence = df.iloc[graph_id]['sequence']
        less_seq = min(len(sequence), args.max_sequence_length)
        for node_id in range(max(0, less_seq - 2)):
            if node_id > 0:
                graphs.add_graph_node_edge(graph_id, node_id, node_id - 1, "Left")
            if node_id < less_seq - 3:
                graphs.add_graph_node_edge(graph_id, node_id, node_id + 1, "Right")
            symbol = sequence[node_id:node_id+3]
            graphs.add_graph_node_property(graph_id, node_id, symbol)

    graphs.encode()
    return graphs, labels
