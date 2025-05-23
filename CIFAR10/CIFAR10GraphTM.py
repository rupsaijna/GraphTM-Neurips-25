from GraphTsetlinMachine.graphs import Graphs
import numpy as np
from scipy.sparse import csr_matrix
from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine
from time import time
import argparse
from skimage.util import view_as_windows
from numba import jit
from keras.datasets import cifar10
import cv2
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def default_args(**kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=30, type=int)
    parser.add_argument("--number-of-clauses", default=80000, type=int)
    parser.add_argument("--T", default=15000, type=int)
    parser.add_argument("--s", default=20.0, type=float)
    parser.add_argument("--number-of-state-bits", default=8, type=int)
    parser.add_argument("--depth", default=1, type=int)
    parser.add_argument("--hypervector-size", default=128, type=int)
    parser.add_argument("--hypervector-bits", default=2, type=int)
    parser.add_argument("--message-size", default=256, type=int)
    parser.add_argument("--message-bits", default=2, type=int)
    parser.add_argument("--patch_size", default=8, type=int)
    parser.add_argument('--double-hashing', dest='double_hashing', default=False, action='store_true')
    parser.add_argument('--one-hot-encoding', dest='one_hot_encoding', default=True, action='store_true')
    parser.add_argument("--max-included-literals", default=32, type=int)
    parser.add_argument("--resolution", default=8, type=int)
    
    args = parser.parse_args(args=[])
    for key, value in kwargs.items():
        if key in args.__dict__:
            setattr(args, key, value)
    return args

args = default_args()

def horizontal_flip(image):
    return cv2.flip(image.astype(np.uint8), 1)

(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

X_train_gaussian = np.copy(X_train)
Y_train = Y_train

X_test_gaussian = np.copy(X_test)
Y_test = Y_test

Y_train = Y_train.reshape(Y_train.shape[0])
Y_test = Y_test.reshape(Y_test.shape[0])

for i in range(X_train.shape[0]):
        for j in range(X_train.shape[3]):
                X_train_gaussian[i,:,:,j] = cv2.adaptiveThreshold(X_train[i,:,:,j], 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2) #cv2.adaptiveThreshold(X_train[i,:,:,j], 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 5)

for i in range(X_test.shape[0]):
        for j in range(X_test.shape[3]):
                X_test_gaussian[i,:,:,j] = cv2.adaptiveThreshold(X_test[i,:,:,j], 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)#cv2.adaptiveThreshold(X_test[i,:,:,j], 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 5)

X_train_gaussian = X_train_gaussian.astype(np.uint32)
X_test_gaussian = X_test_gaussian.astype(np.uint32)
Y_train = Y_train.astype(np.uint32)
Y_test = Y_test.astype(np.uint32)

X_train_thermometer = np.empty(
    (
        X_train.shape[0],
        X_train.shape[1],
        X_train.shape[2],
        X_train.shape[3],
        args.resolution,
    ),
    dtype=np.uint8,
)
for z in range(args.resolution):
    X_train_thermometer[:, :, :, :, z] = X_train[:, :, :, :] >= (z + 1) * 255 / (
        args.resolution + 1
    )

X_test_thermometer = np.empty(
    (
        X_test.shape[0],
        X_test.shape[1],
        X_test.shape[2],
        X_test.shape[3],
        args.resolution,
    ),
    dtype=np.uint8,
)
for z in range(args.resolution):
    X_test_thermometer[:, :, :, :, z] = X_test[:, :, :, :] >= (z + 1) * 255 / (
        args.resolution + 1
    )

X_train_thermometer = X_train_thermometer.reshape(
    (
        X_train.shape[0],
        X_train.shape[1],
        X_train.shape[2],
        3 * args.resolution,
    )
)
X_test_thermometer = X_test_thermometer.reshape(
    (
        X_test.shape[0],
        X_test.shape[1],
        X_test.shape[2],
        3 * args.resolution,
    )
)

dim = 32 - args.patch_size + 1

number_of_nodes = (dim * dim) * 4
print(number_of_nodes)

symbols = []

# Column and row symbols
for i in range(dim):
    symbols.append("C:%d" % (i))
    symbols.append("R:%d" % (i))

# Patch pixel symbols
for i in range(args.patch_size*args.patch_size*3):
    symbols.append(i)

print(symbols)

graphs_train = Graphs(
    X_train.shape[0],
    symbols=symbols,
    hypervector_size=args.hypervector_size,
    hypervector_bits=args.hypervector_bits,
    double_hashing = args.double_hashing,
    one_hot_encoding = args.one_hot_encoding
)

for graph_id in range(X_train.shape[0]):
    graphs_train.set_number_of_graph_nodes(graph_id, number_of_nodes)

graphs_train.prepare_node_configuration()

for graph_id in range(X_train.shape[0]):
    for node_id in range(graphs_train.number_of_graph_nodes[graph_id]):
        node_type_name = "gaussian" if (node_id % 4) < 2 else "thermometer"
        graphs_train.add_graph_node(graph_id, node_id, 0, node_type_name)

graphs_train.prepare_edge_configuration()

for graph_id in range(X_train.shape[0]):
    if graph_id % 1000 == 0:
        print(graph_id, X_train.shape[0])

    image_gaussian = X_train_gaussian[graph_id, :, :]
    flipped_image_gaussian = horizontal_flip(image_gaussian)
     
    windows_gaussian = view_as_windows(image_gaussian, (args.patch_size, args.patch_size, 3))
    flipped_windows_gaussian = view_as_windows(flipped_image_gaussian, (args.patch_size, args.patch_size, 3))
    for q in range(windows_gaussian.shape[0]):
        for r in range(windows_gaussian.shape[1]):
            # Original gaussian node
            node_id_gaussian = (q * dim + r) * 4

            patch_gaussian = windows_gaussian[q,r,0]
            flattened_patch_gaussian = patch_gaussian.reshape(-1).astype(np.uint32)
            for k in flattened_patch_gaussian.nonzero()[0]:
                graphs_train.add_graph_node_property(graph_id, node_id_gaussian, k)
            for s in range(q + 1):
                graphs_train.add_graph_node_property(graph_id, node_id_gaussian, f"C:{s}")
            for s in range(r + 1):
                graphs_train.add_graph_node_property(graph_id, node_id_gaussian, f"R:{s}")

            # Flipped gaussian node
            node_id_gaussian_flipped = (q * dim + r) * 4 + 1

            flipped_patch_gaussian = flipped_windows_gaussian[q, r, 0]
            flattened_flipped_patch_gaussian = flipped_patch_gaussian.reshape(-1).astype(np.uint32)
            for k in flattened_flipped_patch_gaussian.nonzero()[0]:
                graphs_train.add_graph_node_property(graph_id, node_id_gaussian_flipped, k)
            for s in range(q + 1):
                graphs_train.add_graph_node_property(graph_id, node_id_gaussian_flipped, f"C:{s}")
            for s in range(r + 1):
                graphs_train.add_graph_node_property(graph_id, node_id_gaussian_flipped, f"R:{s}")
    
    image_thermometer = X_train_thermometer[graph_id, :, :]
    flipped_image_thermometer = horizontal_flip(image_thermometer)
     
    windows_thermometer = view_as_windows(image_thermometer, (args.patch_size, args.patch_size, 3))
    flipped_windows_thermometer = view_as_windows(flipped_image_thermometer, (args.patch_size, args.patch_size, 3))
    for q in range(windows_thermometer.shape[0]):
        for r in range(windows_thermometer.shape[1]):
            # Original themometers node
            node_id_thermometer = (q * dim + r) * 4 + 2

            patch_thermometer = windows_thermometer[q, r, 0]
            flattened_patch_thermometer = patch_thermometer.reshape(-1).astype(np.uint32)
            for k in flattened_patch_thermometer.nonzero()[0]:
                graphs_train.add_graph_node_property(graph_id, node_id_thermometer, k)
            for s in range(q + 1):
                graphs_train.add_graph_node_property(graph_id, node_id_thermometer, f"C:{s}")
            for s in range(r + 1):
                graphs_train.add_graph_node_property(graph_id, node_id_thermometer, f"R:{s}")

            # Flipped thermometer node
            node_id_thermometer_flipped = (q * dim + r) * 4 + 3

            flipped_patch_thermometer = flipped_windows_thermometer[q, r, 0]
            flattened_flipped_patch_thermometer = flipped_patch_thermometer.reshape(-1).astype(np.uint32)
            for k in flattened_flipped_patch_thermometer.nonzero()[0]:
                graphs_train.add_graph_node_property(graph_id, node_id_thermometer_flipped, k)
            for s in range(q + 1):
                graphs_train.add_graph_node_property(graph_id, node_id_thermometer_flipped, f"C:{s}")
            for s in range(r + 1):
                graphs_train.add_graph_node_property(graph_id, node_id_thermometer_flipped, f"R:{s}")


graphs_train.encode()

print("Training data produced")

graphs_test = Graphs(X_test.shape[0], init_with=graphs_train)
for graph_id in range(X_test.shape[0]):
    graphs_test.set_number_of_graph_nodes(graph_id, number_of_nodes)

graphs_test.prepare_node_configuration()

for graph_id in range(X_test.shape[0]):
    for node_id in range(graphs_test.number_of_graph_nodes[graph_id]):
        node_type_name = "gaussian" if (node_id % 4) < 2 else "thermometer"
        graphs_test.add_graph_node(graph_id, node_id, 0, node_type_name)

graphs_test.prepare_edge_configuration()

for graph_id in range(X_test.shape[0]):
    if graph_id % 1000 == 0:
        print(graph_id, X_test.shape[0])
    
    image_gaussian = X_test_gaussian[graph_id, :, :]
    flipped_image_gaussian = horizontal_flip(image_gaussian)
     
    windows_gaussian = view_as_windows(image_gaussian, (args.patch_size, args.patch_size, 3))
    flipped_windows_gaussian = view_as_windows(flipped_image_gaussian, (args.patch_size, args.patch_size, 3))
    for q in range(windows_gaussian.shape[0]):
        for r in range(windows_gaussian.shape[1]):
            # Original gaussian node
            node_id_gaussian = (q * dim + r) * 4 

            patch_gaussian = windows_gaussian[q,r,0]
            flattened_patch_gaussian = patch_gaussian.reshape(-1).astype(np.uint32)
            for k in flattened_patch_gaussian.nonzero()[0]:
                graphs_test.add_graph_node_property(graph_id, node_id_gaussian, k)
            for s in range(q + 1):
                graphs_test.add_graph_node_property(graph_id, node_id_gaussian, f"C:{s}")
            for s in range(r + 1):
                graphs_test.add_graph_node_property(graph_id, node_id_gaussian, f"R:{s}")

            # Flipped gaussian node
            node_id_gaussian_flipped = (q * dim + r) * 4 + 1

            flipped_patch_gaussian= flipped_windows_gaussian[q, r, 0]
            flattened_flipped_patch_gaussian = flipped_patch_gaussian.reshape(-1).astype(np.uint32)
            for k in flattened_flipped_patch_gaussian.nonzero()[0]:
                graphs_test.add_graph_node_property(graph_id, node_id_gaussian_flipped, k)
            for s in range(q + 1):
                graphs_test.add_graph_node_property(graph_id, node_id_gaussian_flipped, f"C:{s}")
            for s in range(r + 1):
                graphs_test.add_graph_node_property(graph_id, node_id_gaussian_flipped, f"R:{s}")

    image_thermometer = X_test_thermometer[graph_id, :, :]
    flipped_image_thermometer = horizontal_flip(image_thermometer)
     
    windows_thermometer = view_as_windows(image_thermometer, (args.patch_size, args.patch_size, 3))
    flipped_windows_thermometer = view_as_windows(flipped_image_thermometer, (args.patch_size, args.patch_size, 3))
    for q in range(windows_thermometer.shape[0]):
        for r in range(windows_thermometer.shape[1]):
            # Original thermometer node
            node_id_thermometer = (q * dim + r) * 4 + 2

            patch_thermometer = windows_thermometer[q, r, 0]
            flattened_patch_thermometer = patch_thermometer.reshape(-1).astype(np.uint32)

            for k in flattened_patch_thermometer.nonzero()[0]:
                graphs_test.add_graph_node_property(graph_id, node_id_thermometer, k)
            for s in range(q + 1):
                graphs_test.add_graph_node_property(graph_id, node_id_thermometer, f"C:{s}")
            for s in range(r + 1):
                graphs_test.add_graph_node_property(graph_id, node_id_thermometer, f"R:{s}")

            # Flipped thermometer node
            node_id_thermometer_flipped = (q * dim + r) * 4 + 3

            flipped_patch_thermometer = flipped_windows_thermometer[q, r, 0]
            flattened_flipped_patch_thermometer = flipped_patch_thermometer.reshape(-1).astype(np.uint32)
            for k in flattened_flipped_patch_thermometer.nonzero()[0]:
                graphs_test.add_graph_node_property(graph_id, node_id_thermometer_flipped, k)
            for s in range(q + 1):
                graphs_test.add_graph_node_property(graph_id, node_id_thermometer_flipped, f"C:{s}")
            for s in range(r + 1):
                graphs_test.add_graph_node_property(graph_id, node_id_thermometer_flipped, f"R:{s}")


graphs_test.encode()

print("Testing data produced")

tm = MultiClassGraphTsetlinMachine(
    args.number_of_clauses,
    args.T,
    args.s,
    number_of_state_bits = args.number_of_state_bits,
    depth=args.depth,
    message_size=args.message_size,
    message_bits=args.message_bits,
    max_included_literals=args.max_included_literals,
    double_hashing = args.double_hashing,
    one_hot_encoding = args.one_hot_encoding
)

train_accuracies = []
test_accuracies = []

for i in range(args.epochs):    
    start_training = time()
    tm.fit(graphs_train, Y_train, epochs=1, incremental=True)
    stop_training = time()

    start_testing = time()
    result_test = 100*(tm.predict(graphs_test) == Y_test).mean()
    stop_testing = time()

    result_train = 100*(tm.predict(graphs_train) == Y_train).mean()

    train_accuracies.append(result_train)
    test_accuracies.append(result_test)

    print("%d %.2f %.2f %.2f %.2f" % (i, result_train, result_test, stop_training-start_training, stop_testing-start_testing))

