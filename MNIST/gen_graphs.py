import pickle
from lzma import LZMAFile

import numpy as np
from GraphTsetlinMachine.graphs import Graphs
from keras.datasets import mnist
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


def create_graph_subset():
    with LZMAFile("./MNIST/graphs_train.tar.xz", "rb") as f:
        d = pickle.load(f)
    graphs_train: Graphs = d["graph"]

    (_, _), (x_test, y_test) = mnist.load_data()

    x_test = np.where(x_test > 75, 1, 0)
    y_test = y_test.astype(np.uint32)

    # randomly select 1 sample per class
    sample_inds = []

    rng = np.random.default_rng(58)

    for i in range(10):
        inds = np.argwhere(y_test == i).ravel()
        sample_inds.append(rng.choice(inds, 1)[0])

    X = x_test[sample_inds]
    Y = y_test[sample_inds]

    symbols = []
    patch_size = 10
    # Column and row symbols
    for i in range(28 - patch_size + 1):
        symbols.append("C:%d" % (i))
        symbols.append("R:%d" % (i))

    # Patch pixel symbols
    for i in range(patch_size * patch_size):
        symbols.append(i)

    graphs = generate_graphs(
        X,
        dict(
            number_of_graphs=X.shape[0],
            init_with=graphs_train,
        ),
        10,
    )

    return graphs, X, Y


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = np.where(x_train > 75, 1, 0)
    x_test = np.where(x_test > 75, 1, 0)
    y_train = y_train.astype(np.uint32)
    y_test = y_test.astype(np.uint32)

    symbols = []
    patch_size = 10

    # Column and row symbols
    for i in range(28 - patch_size + 1):
        symbols.append("C:%d" % (i))
        symbols.append("R:%d" % (i))

    # Patch pixel symbols
    for i in range(patch_size * patch_size):
        symbols.append(i)

    graphs_train = generate_graphs(
        x_train,
        dict(
            number_of_graphs=x_train.shape[0],
            symbols=symbols,
            hypervector_size=128,
            hypervector_bits=2,
            double_hashing=False,
        ),
        patch_size,
    )

    graphs_test = generate_graphs(
        x_test,
        dict(
            number_of_graphs=x_test.shape[0],
            init_with=graphs_train,
        ),
        patch_size,
    )

    print("Graphs generated")

    with LZMAFile("./MNIST/graphs_train.tar.lz", "wb") as f:
        pickle.dump({"graph": graphs_train, "y": y_train}, f)

    with LZMAFile("./MNIST/graphs_test.tar.lz", "wb") as f:
        pickle.dump({"graph": graphs_test, "y": y_test}, f)

    print("Graphs saved to ./MNIST/graphs_train.tar.lz and ./MNIST/graphs_test.tar.lz")
