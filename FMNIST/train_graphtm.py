import pickle
from lzma import LZMAFile
from time import time

import numpy as np
from GraphTsetlinMachine.graphs import Graphs
from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine


if __name__ == "__main__":
    with LZMAFile("./FMNIST/graphs_train.tar.lz", "rb") as f:
        dtrain = pickle.load(f)
    with LZMAFile("./FMNIST/graphs_test.tar.lz", "rb") as f:
        dtest = pickle.load(f)

    graphs_train: Graphs = dtrain["graph"]
    y_train = dtrain["y"]

    graphs_test: Graphs = dtest["graph"]
    y_test = dtest["y"]

    tm = MultiClassGraphTsetlinMachine(40000, 15000, 10, depth=1)

    for i in range(50):
        start_training = time()
        tm.fit(graphs_train, y_train, epochs=1, incremental=True)
        stop_training = time()

        start_testing = time()
        result_test = 100 * (tm.predict(graphs_test) == y_test).mean()
        stop_testing = time()

        result_train = 100 * (tm.predict(graphs_train) == y_train).mean()

        print(
            f"Epoch {i} | Train Acc: {result_train:.4f}, Test Acc: {result_test:.4f} | Train Time: {stop_training - start_training:.2f}, Test Time: {stop_testing - start_testing:.2f}"
        )

    state_dict = tm.save()

    with LZMAFile("./FMNIST/fmnist_model.tm", "wb") as f:
        pickle.dump(state_dict, f)
