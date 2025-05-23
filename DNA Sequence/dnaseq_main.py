import pandas as pd
import numpy as np
import time
from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine
from sklearn.model_selection import train_test_split
from utils import (
    filter_valid_sequences, get_sequence_length_stats, 
    get_unique_bases_per_label, sample_balanced_data, 
    create_graph_data
)

# Load data
df = pd.read_csv('data/Sequence_Dataset.csv')
filtered_data = filter_valid_sequences(df)

print(f"Original dataset size: {len(df)}")
print(f"Filtered dataset size: {len(filtered_data)}")

print(filtered_data['label'].value_counts())

max_len, min_len = get_sequence_length_stats(filtered_data)
print("Max lengths:", max_len)
print("Min lengths:", min_len)

print(get_unique_bases_per_label(filtered_data))

labels = ["SARS-CoV-2", "Influenza virus", "Dengue virus", "Zika virus", "Rotavirus"]
sampled_df = sample_balanced_data(filtered_data, labels)
print(sampled_df['label'].value_counts())

train_df, test_df = train_test_split(sampled_df, test_size=0.2, random_state=42)

label_mapping = {
    "SARS-CoV-2": 0,
    "Influenza virus": 1,
    "Dengue virus": 2,
    "Rotavirus": 3,
    "Zika virus": 4
}

class Args:
    number_of_examples_train = len(train_df)
    number_of_examples_test = len(test_df)
    number_of_classes = 5
    max_sequence_length = 500
    hypervector_size = 512
    hypervector_bits = 2
    double_hashing = False
    message_size = 512
    message_bits = 2
    number_of_state_bits = 8

args = Args()

print("Creating training data")
graphs_train, Y_train = create_graph_data(train_df, args, label_mapping, is_train=True)
print("TRAIN DATA CREATED")

print("Creating test data")
graphs_test, Y_test = create_graph_data(test_df, args, label_mapping, is_train=False, init_graph=graphs_train)
print("TEST DATA CREATED")

tm = MultiClassGraphTsetlinMachine(
    T=2000,
    s=1.0,
    depth=2,
    number_of_clauses=2000,
    max_included_literals=200
)

for i in range(20):
    start_training = time.time()
    tm.fit(graphs_train, Y_train, epochs=1, incremental=True)
    stop_training = time.time()

    start_testing = time.time()
    result_test = 100 * (tm.predict(graphs_test) == Y_test).mean()
    stop_testing = time.time()

    result_train = 100 * (tm.predict(graphs_train) == Y_train).mean()

    print("%d %.2f %.2f %.2f %.2f" % (i, result_train, result_test, stop_training - start_training, stop_testing - start_testing))
