from lzma import LZMAFile
import pickle

from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import numpy as np
from GraphTsetlinMachine.graphs import Graphs
from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine
from seaborn import color_palette
from tqdm import tqdm
from gen_graphs import generate_graphs
from keras.datasets import fashion_mnist

icefire = color_palette("icefire", as_cmap=True)


def create_graph_subset():
    with LZMAFile("./FMNIST/graphs_train.tar.lz", "rb") as f:
        d = pickle.load(f)
    graphs_train: Graphs = d["graph"]

    (_, _), (x_test_org, y_test) = fashion_mnist.load_data()

    resolution = 8
    x_test = np.empty((*x_test_org.shape, resolution), dtype=np.uint8)

    # Quantize pixel values
    for z in range(resolution):
        threshold = (z + 1) * 255 / (resolution + 1)
        x_test[..., z] = (x_test_org >= threshold) & 1
    y_test = y_test.astype(np.uint32)

    # randomly select 1 sample per class
    sample_inds = []

    rng = np.random.default_rng(75)

    for i in range(10):
        inds = np.argwhere(y_test == i).ravel()
        sample_inds.append(rng.choice(inds, 1)[0])

    X = x_test[sample_inds]
    Y = y_test[sample_inds]

    graphs = generate_graphs(
        X,
        dict(
            number_of_graphs=X.shape[0],
            init_with=graphs_train,
        ),
        3,
    )

    class_names = [
        "tshirt",
        "trouser",
        "pullover",
        "dress",
        "coat",
        "sandal",
        "shirt",
        "sneaker",
        "bag",
        "ankleboot",
    ]

    return graphs, X, Y, class_names


def unbinarize(a, levels):
    def f(b):
        nz = np.argwhere(b).ravel()
        return nz[-1] + 1 if len(nz) > 0 else 0

    return np.apply_along_axis(f, -1, a) / levels


def transform(tm: MultiClassGraphTsetlinMachine, graphs: Graphs, X, Y):
    weights = tm.get_state()[1].reshape(tm.number_of_outputs, tm.number_of_clauses)
    clause_literals = tm.get_clause_literals(graphs.hypervectors)
    total_symbols = len(graphs.symbol_id)
    patch_size = 3
    position_symbols = 28 - patch_size + 1
    levels = 8

    co_nodewise, class_sums = tm.transform_nodewise(graphs)
    co_nodewise = co_nodewise.reshape((X.shape[0], tm.number_of_clauses, position_symbols, position_symbols))

    positive_literals = clause_literals[:, 2 * position_symbols : total_symbols].reshape(
        (tm.number_of_clauses, patch_size, patch_size, levels)
    )
    negative_literals = clause_literals[:, total_symbols + 2 * position_symbols :].reshape(
        (tm.number_of_clauses, patch_size, patch_size, levels)
    )

    positive_literals = unbinarize(positive_literals, levels).squeeze()
    negative_literals = unbinarize(negative_literals, levels).squeeze()

    transformed = np.zeros((X.shape[0], 2, 28, 28))
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


def plot_clauses(transformed, X, class_names):
    levels = 8
    mosiac = """
    a0.b1.c2.d3.e4C
    f5.g6.h7.i8.j9C
    """
    fig, axd = plt.subplot_mosaic(
        mosiac,
        layout="compressed",
        figsize=(25, 7),
        width_ratios=[1, 1, 0.2, 1, 1, 0.2, 1, 1, 0.2, 1, 1, 0.2, 1, 1, 0.1],
    )

    for _, ax in axd.items():
        ax.axis("off")

    for e in range(10):
        clause_image = transformed[e, 0] - transformed[e, 1]
        clause_image[clause_image < 0] = clause_image[clause_image < 0] / (-1 * np.min(clause_image) + 1e-7)
        clause_image[clause_image > 0] = clause_image[clause_image > 0] / (np.max(clause_image) + 1e-7)

        normed = Normalize(-1, 1)(clause_image)

        axd[chr(e + ord("a"))].imshow(unbinarize(X[e], levels), cmap="gray")
        axd[f"{e}"].imshow(normed, cmap=icefire)
        axd[chr(e + ord("a"))].set_title(f"({chr(e + ord('a'))})", loc="right")
        axd[f"{e}"].set_title(class_names[e], loc="left")

    fig.colorbar(axd["0"].images[0], ax=axd["C"], fraction=1, pad=0)

    return fig, axd


if __name__ == "__main__":
    graphs, X, Y, class_names = create_graph_subset()

    with LZMAFile("./FMNIST/fmnist_model.tm", "rb") as f:
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
    fig, axd = plot_clauses(transformed, X, class_names)
    fig.savefig("./FMNIST/FMNIST_clauses.png", dpi=300, bbox_inches="tight")
