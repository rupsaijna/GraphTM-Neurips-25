import pickle
from lzma import LZMAFile

import numpy as np
from GraphTsetlinMachine.graphs import Graphs
from keras.datasets import fashion_mnist
from tqdm import tqdm


def generate_graphs(X, graph_args, patch_size):
    graphs = Graphs(**graph_args)

    num_graphs = X.shape[0]
    dim = X.shape[1] - patch_size + 1
    num_nodes = dim * dim

    for id in range(num_graphs):
        graphs.set_number_of_graph_nodes(id, num_nodes)

    graphs.prepare_node_configuration()

    for graph_id in tqdm(range(X.shape[0]), desc="Adding graph nodes", leave=False):
        for node_id in range(graphs.number_of_graph_nodes[graph_id]):
            graphs.add_graph_node(graph_id, node_id, 0)

    graphs.prepare_edge_configuration()

    for graph_id in tqdm(range(X.shape[0]), desc="Adding node symbols", leave=False):

        for node_id in range(num_nodes):
            x, y = node_id // dim, node_id % dim
            patch = X[graph_id, x : x + patch_size, y : y + patch_size].flatten()

            graphs.add_graph_node_property(graph_id, node_id, "R:%d" % (x))
            graphs.add_graph_node_property(graph_id, node_id, "C:%d" % (y))

            for p in patch.nonzero()[0]:
                graphs.add_graph_node_property(graph_id, node_id, p)

    graphs.encode()

    return graphs


if __name__ == "__main__":
    (x_train_org, y_train), (x_test_org, y_test) = fashion_mnist.load_data()

    resolution = 8
    x_train = np.empty((*x_train_org.shape, resolution), dtype=np.uint8)
    x_test = np.empty((*x_test_org.shape, resolution), dtype=np.uint8)

    # Quantize pixel values
    for z in range(resolution):
        threshold = (z + 1) * 255 / (resolution + 1)
        x_train[..., z] = (x_train_org >= threshold) & 1
        x_test[..., z] = (x_test_org >= threshold) & 1

    y_train = y_train.astype(np.uint32)
    y_test = y_test.astype(np.uint32)

    symbols = []
    patch_size = 3

    # Column and row symbols
    for i in range(28 - patch_size + 1):
        symbols.append("C:%d" % (i))
        symbols.append("R:%d" % (i))

    # Patch pixel symbols
    for i in range(patch_size * patch_size * resolution):
        symbols.append(i)

    graphs_train = generate_graphs(
        x_train_org,
        dict(
            number_of_graphs=x_train_org.shape[0],
            symbols=symbols,
            hypervector_size=256,
            hypervector_bits=2,
            double_hashing=True,
        ),
        patch_size,
    )

    graphs_test = generate_graphs(
        x_test_org,
        dict(
            number_of_graphs=x_test_org.shape[0],
            init_with=graphs_train,
        ),
        patch_size,
    )

    print("Graphs generated")

    with LZMAFile("./FMNIST/graphs_train.tar.lz", "wb") as f:
        pickle.dump({"graph": graphs_train, "y": y_train}, f)

    with LZMAFile("./FMNIST/graphs_test.tar.lz", "wb") as f:
        pickle.dump({"graph": graphs_test, "y": y_test}, f)

    print("Graphs saved to ./FMNIST/graphs_train.tar.lz and ./FMNIST/graphs_test.tar.lz")

