import pickle
from lzma import LZMAFile

import matplotlib.pyplot as plt
import numpy as np
from GraphTsetlinMachine.graphs import Graphs
from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine
from keras.api.datasets import mnist
from matplotlib.colors import Normalize
from seaborn import color_palette
from tqdm import tqdm

from gen_graphs import generate_graphs

icefire = color_palette("icefire", as_cmap=True)


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


def transform(tm: MultiClassGraphTsetlinMachine, graphs: Graphs, X, Y):
    weights = tm.get_state()[1].reshape(tm.number_of_outputs, tm.number_of_clauses)
    clause_literals = tm.get_clause_literals(graphs.hypervectors)
    total_symbols = len(graphs.symbol_id)
    patch_size = 10
    position_symbols = 28 - patch_size + 1

    co_nodewise, class_sums = tm.transform_nodewise(graphs)
    co_nodewise = co_nodewise.reshape((X.shape[0], tm.number_of_clauses, position_symbols, position_symbols))

    positive_literals = clause_literals[:, 2 * position_symbols : total_symbols].reshape(
        (tm.number_of_clauses, patch_size, patch_size)
    )
    negative_literals = clause_literals[:, total_symbols + 2 * position_symbols :].reshape(
        (tm.number_of_clauses, patch_size, patch_size)
    )

    transformed = np.zeros((10, 2, 28, 28))
    for e in range(X.shape[0]):
        for ci in tqdm(range(tm.number_of_clauses), desc=f"Processing {e}", leave=False):
            timg = np.zeros((2, 28, 28))
            if weights[Y[e], ci] > 0:
                active_pos = np.argwhere(co_nodewise[e, ci])

                for m, n in active_pos:
                    timg[0, m : m + patch_size, n : n + patch_size] += positive_literals[ci]
                    timg[1, m : m + patch_size, n : n + patch_size] += negative_literals[ci]

                transformed[e] += timg * weights[Y[e], ci]

    return transformed

def plot_clauses(transformed, X):
    mosiac = """
    a0.b1.c2.d3.e4C
    f5.g6.h7.i8.j9C
    """
    fig, axd = plt.subplot_mosaic(
        mosiac,
        layout="compressed",
        figsize=(15, 5),
        width_ratios=[1, 1, 0.2, 1, 1, 0.2, 1, 1, 0.2, 1, 1, 0.2, 1, 1, 0.1],
    )

    for _, ax in axd.items():
        ax.axis("off")

    for e in range(10):
        clause_image = transformed[e, 0] - transformed[e, 1]
        clause_image[clause_image < 0] = clause_image[clause_image < 0] / (-1 * np.min(clause_image) + 1e-7)
        clause_image[clause_image > 0] = clause_image[clause_image > 0] / (np.max(clause_image) + 1e-7)

        normed = Normalize(-1, 1)(clause_image)

        axd[chr(e + ord("a"))].imshow(X[e], cmap="gray")
        axd[f"{e}"].imshow(normed, cmap=icefire)

    fig.colorbar(axd["0"].images[0], ax=axd["C"], fraction=1, pad=0)

    return fig, axd


if __name__ == "__main__":
    graphs, X, Y = create_graph_subset()

    with LZMAFile("./MNIST/mnist_model.tm", "rb") as f:
        state_dict = pickle.load(f)

    params = dict(
        number_of_clauses=state_dict["number_of_clauses"],
        T=state_dict["T"],
        s=state_dict["s"],
        message_size=state_dict["message_size"],
        message_bits=state_dict["message_bits"],
    )

    tm = MultiClassGraphTsetlinMachine(**params)
    tm.load(state_dict)

    transformed = transform(tm, graphs, X, Y)
    fig, axd = plot_clauses(transformed, X)
    fig.savefig("./MNIST/clauses.png", dpi=300, bbox_inches="tight")
