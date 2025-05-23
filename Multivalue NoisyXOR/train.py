import argparse
import random
from time import time
import numpy as np
from GraphTsetlinMachine.graphs import Graphs
from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine


def default_args(**kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--s", default=2.2, type=float)
    parser.add_argument("--number-of-state-bits", default=8, type=int)
    parser.add_argument("--q", default=1.0, type=float)
    parser.add_argument("--depth", default=2, type=int)
    parser.add_argument("--hypervector-size", default=2048, type=int)
    parser.add_argument("--hypervector-bits", default=2, type=int)
    parser.add_argument("--message-bits", default=2, type=int)
    parser.add_argument('--double-hashing', dest='double_hashing', default=False, action='store_true')
    parser.add_argument("--noise", default=0.01, type=float)
    parser.add_argument("--number-of-examples", default=50000, type=int)
    parser.add_argument("--number-of-trials", default=5, type=int)
    parser.add_argument("--number-of-values", default=500, type=int)

    args = parser.parse_args()
    for key, value in kwargs.items():
        if key in args.__dict__:
            setattr(args, key, value)
    return args


args = default_args()
number_of_test_examples = args.number_of_examples//10

number_of_clauses_range = [4, 50, 200, 1000, 2000]
hv_size_range = [int(2**i) for i in range(11, 14)]


symbols = []
for value in range(args.number_of_values):
    symbols.append(value)

# ======================
# Create Training data
# ======================
graphs_train = Graphs(
    args.number_of_examples,
    symbols=symbols,
    hypervector_size=args.hypervector_size,
    hypervector_bits=args.hypervector_bits,
    double_hashing = args.double_hashing
)

for graph_id in range(args.number_of_examples):
    graphs_train.set_number_of_graph_nodes(graph_id, 2)

graphs_train.prepare_node_configuration()

for graph_id in range(args.number_of_examples):
    for node_id in range(graphs_train.number_of_graph_nodes[graph_id]):
        number_of_edges = 1
        graphs_train.add_graph_node(graph_id, node_id, number_of_edges)

graphs_train.prepare_edge_configuration()

Y_train = np.empty(args.number_of_examples, dtype=np.uint32)
for graph_id in range(args.number_of_examples):
    edge_type = "Plain"
    source_node_id = 0
    destination_node_id = 1
    graphs_train.add_graph_node_edge(graph_id, source_node_id, destination_node_id, edge_type)

    source_node_id = 1
    destination_node_id = 0
    graphs_train.add_graph_node_edge(graph_id, source_node_id, destination_node_id, edge_type)

    x1 = random.choice(symbols)
    x2 = random.choice(symbols)
    if (x1 % 2) == (x2 % 2):
        Y_train[graph_id] = 0
    else:
        Y_train[graph_id] = 1

    graphs_train.add_graph_node_property(graph_id, 0, x1)
    graphs_train.add_graph_node_property(graph_id, 1, x2)

    if np.random.rand() <= args.noise:
        Y_train[graph_id] = 1 - Y_train[graph_id]

graphs_train.encode()


# ================
# Create Test data
# ================
print("Creating testing data")
graphs_test = Graphs(number_of_test_examples, init_with=graphs_train)

for graph_id in range(number_of_test_examples):
    graphs_test.set_number_of_graph_nodes(graph_id, 2)

graphs_test.prepare_node_configuration()
for graph_id in range(number_of_test_examples):
    for node_id in range(graphs_test.number_of_graph_nodes[graph_id]):
        number_of_edges = 1
        graphs_test.add_graph_node(graph_id, node_id, number_of_edges)

graphs_test.prepare_edge_configuration()
Y_test = np.empty(number_of_test_examples, dtype=np.uint32)
for graph_id in range(number_of_test_examples):
    edge_type = "Plain"
    source_node_id = 0
    destination_node_id = 1
    graphs_test.add_graph_node_edge(graph_id, source_node_id, destination_node_id, edge_type)

    source_node_id = 1
    destination_node_id = 0
    graphs_test.add_graph_node_edge(graph_id, source_node_id, destination_node_id, edge_type)

    x1 = random.choice(symbols)
    x2 = random.choice(symbols)
    if (x1 % 2) == (x2 % 2):
        Y_test[graph_id] = 0
    else:
        Y_test[graph_id] = 1

    graphs_test.add_graph_node_property(graph_id, 0, x1)
    graphs_test.add_graph_node_property(graph_id, 1, x2)

graphs_test.encode()


# ==========
# Run models
# ==========
for number_of_clauses in reversed(number_of_clauses_range):
    for msg_size in hv_size_range:
        # Create file
        result_file = f"results_unique_symbols_exp_{number_of_clauses}_{msg_size}.csv"
    
        T = int(10*number_of_clauses)
        
        with open(result_file, 'w') as f:
            f.write("trial,number of clauses,message size,epoch,accuracy\n")
        
        print("Training TM")
        for r in range(args.number_of_trials):

            tm = MultiClassGraphTsetlinMachine(
                number_of_clauses,
                T,
                args.s,
                number_of_state_bits = args.number_of_state_bits,
                depth = args.depth,
                message_size = msg_size,
                message_bits = args.message_bits,
                max_included_literals = None,
                double_hashing = args.double_hashing,
                grid = (16*13,1,1),
                block = (128,1,1),
            )
            tic = time()
            for i in range(args.epochs):
                tm.fit(graphs_train, Y_train, epochs=1, incremental=True)
                accuracy = 100*(tm.predict(graphs_test) == Y_test).mean()
        
                with open(result_file, 'a') as f:
                    f.write(f"{r},{number_of_clauses},{msg_size},{i},{accuracy}\n")
                    
            toc = time()
            print(f"Trial {r+1} done in {tic - toc:2f} seconds")
