
import numpy as np
import argparse
from tensorflow.keras.datasets import imdb
from GraphTsetlinMachine.graphs import Graphs
from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine
from time import time
from collections import Counter
import random
import nltk

nltk.download('punkt')
# Argument parser setup
def default_args(**kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--number-of-clauses", default=10000, type=int)
    parser.add_argument("--T", default=100000, type=int)
    parser.add_argument("--s", default=15.0, type=float)
    parser.add_argument("--number-of-state-bits", default=1024, type=int)
    parser.add_argument("--depth", default=1, type=int)
    parser.add_argument("--hypervector-size", default=2048, type=int)
    parser.add_argument("--hypervector-bits", default=1024, type=int)
    parser.add_argument("--message-size", default=2048, type=int)
    parser.add_argument("--message-bits", default=1024, type=int)
    parser.add_argument('--double-hashing', dest='double_hashing', default=False, action='store_true')
    parser.add_argument("--max-included-literals", default=None, type=int)
    parser.add_argument("--imdb_num_words", default=10000, type=int)  
    parser.add_argument("--imdb_index_from", default=2, type=int)
    parser.add_argument("--seed", default=42, type=int, help="Random seed for reproducibility")
    parser.add_argument("--num-runs", default=2, type=int)

    args = parser.parse_args()
    for key, value in kwargs.items():
        if key in args.__dict__:
            setattr(args, key, value)
    return args

def calculate_class_metrics(y_true, y_pred):
    """Calculate per-class metrics and their standard deviations"""
    # Get confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    
    # Calculate metrics
    total = len(y_true)
    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Calculate class-specific accuracies
    negative_acc = tn / (tn + fp) if (tn + fp) > 0 else 0
    positive_acc = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    # Calculate standard deviations for binary predictions
    acc_std = np.sqrt((accuracy * (1 - accuracy)) / total)
    neg_acc_std = np.sqrt((negative_acc * (1 - negative_acc)) / (tn + fp)) if (tn + fp) > 0 else 0
    pos_acc_std = np.sqrt((positive_acc * (1 - positive_acc)) / (tp + fn)) if (tp + fn) > 0 else 0
    
    return {
        'accuracy': accuracy * 100,
        'accuracy_std': acc_std * 100,
        'precision': precision * 100,
        'recall': recall * 100,
        'f1': f1 * 100,
        'negative_acc': negative_acc * 100,
        'negative_acc_std': neg_acc_std * 100,
        'positive_acc': positive_acc * 100,
        'positive_acc_std': pos_acc_std * 100,
        'confusion_matrix': {
            'tn': tn,
            'fp': fp,
            'fn': fn,
            'tp': tp
        }
    }

def count_possible_edges(node_id, sequence_length):
    """Count possible edges for a node based on position."""
    possible_edges = 0
    for offset in range(1, 5):
        if node_id + offset < sequence_length: possible_edges += 1  # forward edges
        if node_id - offset >= 0: possible_edges += 1              # backward edges
    return possible_edges

def configure_graphs(graphs, data_x, id_to_word, label="train"):
    # Set number of nodes for each graph
    for graph_id, review in enumerate(data_x):
        # Ensure at least one node
        text_length = max(1, len(review))
        graphs.set_number_of_graph_nodes(graph_id, text_length)
    
    graphs.prepare_node_configuration()
    
    # Configure nodes and their edges
    for graph_id, review in enumerate(data_x):
        if len(review) <= 0:
            # Special case: empty text, add placeholder node with no edges
            node_name = "Node_0"
            graphs.add_graph_node(graph_id, node_name, 0)
        else:
            for node_id in range(len(review)):
                possible_edges = count_possible_edges(node_id, len(review))
                graphs.add_graph_node(graph_id, f"Node_{node_id}", possible_edges)
    
    graphs.prepare_edge_configuration()
    
    # Add edges and node properties
    for graph_id, review in enumerate(data_x):
        if len(review) <= 1:
            # For single-token texts, don't add edges
            if len(review) == 1:
                word_id = review[0]
                if word_id > 0:  # Skip padding tokens
                    word = id_to_word.get(word_id, f"UNK_{word_id}")
                    graphs.add_graph_node_property(graph_id, "Node_0", word)
        else:
            for node_id in range(len(review)):
                # Add edges with limited offset
                max_offset = min(4, len(review) - 1)  # Limit offset to review length
                
                for offset in range(1, max_offset + 1):
                    # Backward edges
                    if node_id - offset >= 0:
                        graphs.add_graph_node_edge(graph_id, f"Node_{node_id}", f"Node_{node_id - offset}", f"-{offset}")
                    
                    # Forward edges
                    if node_id + offset < len(review):
                        graphs.add_graph_node_edge(graph_id, f"Node_{node_id}", f"Node_{node_id + offset}", f"+{offset}")
                
                # Add word property to node - use the word itself as the symbol
                word_id = review[node_id]
                if word_id > 0:  # Skip padding tokens
                    word = id_to_word.get(word_id, f"UNK_{word_id}")
                    graphs.add_graph_node_property(graph_id, f"Node_{node_id}", word)
    
    # Debug information before encoding
    print(f"Graph structure before encoding:")
    print(f"Number of graphs: {len(data_x)}")
    
    try:
        graphs.encode()
        print(f"Graph construction completed for {label} data")
        print(f"Number of nodes in graph: {graphs.number_of_nodes}")
        print(f"Edge index size: {len(graphs.edge_index) if hasattr(graphs, 'edge_index') else 'N/A'}")
    except Exception as e:
        print(f"Error during graph encoding: {str(e)}")
        raise

def run_single_experiment(args, run_id=0):
    """Run a single experiment with the given arguments"""
    print(f"\n{'='*20} EXPERIMENT RUN {run_id+1} {'='*20}\n")
    
    # Set a different random seed for each run
    random_seed = args.seed + run_id
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # Load dataset
    print("Preparing dataset")
    (train_x, train_y), (test_x, test_y) = imdb.load_data(num_words=args.imdb_num_words, index_from=args.imdb_index_from)

    # Map words to IDs and create reverse mapping
    word_to_id = imdb.get_word_index()
    word_to_id = {k: (v + args.imdb_index_from) for k, v in word_to_id.items()}
    word_to_id["<PAD>"] = 0
    word_to_id["<START>"] = 1
    word_to_id["<UNK>"] = 2

    id_to_word = {value: key for key, value in word_to_id.items()}

    # Create symbols - just use the words themselves
    symbols = [id_to_word.get(i, f"UNK_{i}") for i in range(args.imdb_num_words + args.imdb_index_from)]
    print(f"Created {len(symbols)} symbols")

    # Calculate text lengths for analysis
    train_lengths = np.array([len(review) for review in train_x])
    test_lengths = np.array([len(review) for review in test_x])
    
    # Output length distribution information
    print("\nText Length Distribution (in words):")
    for threshold in [50, 100, 200, 300, 500]:
        count = np.sum(train_lengths < threshold)
        print(f"Reviews < {threshold} words: {count} ({count/len(train_x)*100:.1f}%)")

    # Initialize training graphs
    print("\nInitializing training graphs...")
    graphs_train = Graphs(
        len(train_x),
        symbols=symbols,
        hypervector_size=args.hypervector_size,
        hypervector_bits=args.hypervector_bits
    )
    
    # Configure training graphs
    configure_graphs(graphs_train, train_x, id_to_word, "training")
    
    # Initialize and configure testing graphs
    print("\nInitializing testing graphs...")
    graphs_test = Graphs(len(test_x), init_with=graphs_train)
    configure_graphs(graphs_test, test_x, id_to_word, "testing")
    
    # Initialize Tsetlin Machine
    print("\nInitializing Tsetlin Machine...")
    tm = MultiClassGraphTsetlinMachine(
        args.number_of_clauses,
        args.T,
        args.s,
        number_of_state_bits=args.number_of_state_bits,
        depth=args.depth,
        message_size=args.message_size,
        message_bits=args.message_bits,
        max_included_literals=args.max_included_literals,
        double_hashing=args.double_hashing,
        grid=(16 * 13 * 4, 1, 1),
        block=(128, 1, 1)
    )
    
    # Arrays to store accuracy at each epoch
    train_accuracies = []
    test_accuracies = []
    
    # Training loop
    print("\nStarting training...")
    best_test_acc = 0.0
    best_epoch = 0
    bootstrap_test_accuracies = []
    
    for epoch in range(args.epochs):
        start_training = time()
        tm.fit(graphs_train, train_y, epochs=1, incremental=True)
        stop_training = time()

        start_testing = time()
        train_pred = tm.predict(graphs_train)
        test_pred = tm.predict(graphs_test)
        train_acc = 100 * (train_pred == train_y).mean()
        test_acc = 100 * (test_pred == test_y).mean()
        
        # Store accuracies for standard deviation calculation
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        
        # Calculate standard deviations
        if len(train_accuracies) > 1:
            train_acc_std = np.std(train_accuracies)
            test_acc_std = np.std(test_accuracies)
            
            # Calculate recent standard deviation (last 5 epochs)
            recent_train_std = np.std(train_accuracies[-min(5, len(train_accuracies)):])
            recent_test_std = np.std(test_accuracies[-min(5, len(test_accuracies)):])
        else:
            train_acc_std = test_acc_std = recent_train_std = recent_test_std = 0.0
        
        # Track best model
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch + 1
            best_test_pred = test_pred.copy()
            
            # Bootstrap for confidence intervals on best model
            bootstrap_samples = 1000
            bootstrap_test_accuracies = []
            for _ in range(bootstrap_samples):
                # Generate bootstrap sample indices
                indices = np.random.choice(len(test_y), len(test_y), replace=True)
                bootstrap_acc = 100 * (test_pred[indices] == test_y[indices]).mean()
                bootstrap_test_accuracies.append(bootstrap_acc)
            
        stop_testing = time()

        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print(f"Train Accuracy: {train_acc:.2f}% (StdDev: {train_acc_std:.2f}%, Recent StdDev: {recent_train_std:.2f}%)")
        print(f"Test Accuracy: {test_acc:.2f}% (StdDev: {test_acc_std:.2f}%, Recent StdDev: {recent_test_std:.2f}%)")
        print(f"Best Test Accuracy: {best_test_acc:.2f}% (Epoch {best_epoch})")
        print(f"Training Time: {stop_training - start_training:.2f}s, Testing Time: {stop_testing - start_testing:.2f}s")
    
    
    # Calculate final standard deviations
    final_train_acc_std = np.std(train_accuracies)
    final_test_acc_std = np.std(test_accuracies)
    

    if bootstrap_test_accuracies:
        bootstrap_test_accuracies.sort()
        lower_ci = bootstrap_test_accuracies[int(0.025 * len(bootstrap_test_accuracies))]
        upper_ci = bootstrap_test_accuracies[int(0.975 * len(bootstrap_test_accuracies))]
    else:
        lower_ci = upper_ci = best_test_acc
    
    # After all epochs, print summary statistics
    print(f"RUN {run_id+1} FINAL RESULTS")
    print(f"Best Test Accuracy: {best_test_acc:.2f}% (Epoch {best_epoch})")
    print(f"95% Confidence Interval: [{lower_ci:.2f}%, {upper_ci:.2f}%]")
    print(f"Standard Deviations:")
    print(f"  - Train Accuracy: {final_train_acc_std:.2f}%")
    print(f"  - Test Accuracy: {final_test_acc_std:.2f}%")
    
    # Return final metrics for aggregation
    return {
        'best_test_acc': best_test_acc,
        'best_epoch': best_epoch,
        'train_accuracies': train_accuracies,
        'test_accuracies': test_accuracies,
        'train_acc_std': final_train_acc_std,
        'test_acc_std': final_test_acc_std,
        'confidence_interval': (lower_ci, upper_ci),
        'best_test_pred': best_test_pred,
        'id_to_word': id_to_word,
        'test_x': test_x,
        'test_y': test_y
    }
    
def main():
    args = default_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # If multiple runs requested, perform them and calculate statistics across runs
    if args.num_runs > 1:
        print(f"\nPerforming {args.num_runs} experimental runs for standard deviation calculation...\n")
        
        # Store results from each run
        all_results = []
        best_test_accs = []
        final_train_accs = []
        final_test_accs = []
        
        # Run each experiment
        for run_id in range(args.num_runs):
            results = run_single_experiment(args, run_id)
            all_results.append(results)
            
            # Extract key metrics
            best_test_accs.append(results['best_test_acc'])
            if results['train_accuracies']:
                final_train_accs.append(results['train_accuracies'][-1])
            if results['test_accuracies']:
                final_test_accs.append(results['test_accuracies'][-1])
        
        # Calculate statistics across runs
        best_acc_mean = np.mean(best_test_accs)
        best_acc_std = np.std(best_test_accs)
        
        final_train_mean = np.mean(final_train_accs) if final_train_accs else 0
        final_train_std = np.std(final_train_accs) if final_train_accs else 0
        
        final_test_mean = np.mean(final_test_accs) if final_test_accs else 0
        final_test_std = np.std(final_test_accs) if final_test_accs else 0
        
        # Calculate epoch-wise statistics if all runs had the same number of epochs
        epoch_train_means = []
        epoch_train_stds = []
        epoch_test_means = []
        epoch_test_stds = []
        
        min_epochs = min(len(results['train_accuracies']) for results in all_results)
        
        for epoch in range(min_epochs):
            epoch_train_accs = [results['train_accuracies'][epoch] for results in all_results]
            epoch_test_accs = [results['test_accuracies'][epoch] for results in all_results]
            
            epoch_train_means.append(np.mean(epoch_train_accs))
            epoch_train_stds.append(np.std(epoch_train_accs))
            epoch_test_means.append(np.mean(epoch_test_accs))
            epoch_test_stds.append(np.std(epoch_test_accs))
        
        # Print aggregated results
        print("\n" + "="*60)
        print(f"AGGREGATED RESULTS OVER {args.num_runs} RUNS")
        print("="*60)
        print(f"Best Test Accuracy: {best_acc_mean:.2f}% ± {best_acc_std:.2f}%")
        print(f"Final Train Accuracy: {final_train_mean:.2f}% ± {final_train_std:.2f}%")
        print(f"Final Test Accuracy: {final_test_mean:.2f}% ± {final_test_std:.2f}%")
        
        # Print epoch-wise statistics
        print("\nEpoch-wise Statistics:")
        print(f"{'Epoch':<6} {'Train Acc Mean':<15} {'Train Acc Std':<15} {'Test Acc Mean':<15} {'Test Acc Std':<15}")
        print("-" * 70)
        
        for epoch in range(min_epochs):
            print(f"{epoch+1:<6} {epoch_train_means[epoch]:<15.2f} {epoch_train_stds[epoch]:<15.2f} "
                  f"{epoch_test_means[epoch]:<15.2f} {epoch_test_stds[epoch]:<15.2f}")
        
        # Print standard deviation stability
        avg_train_std = np.mean([results['train_acc_std'] for results in all_results])
        avg_test_std = np.mean([results['test_acc_std'] for results in all_results])
        
        print("\nAverage Within-Run Standard Deviations:")
        print(f"Train Accuracy: {avg_train_std:.2f}%")
        print(f"Test Accuracy: {avg_test_std:.2f}%")
        
        print("\nBetween-Run Standard Deviations:")
        print(f"Best Test Accuracy: {best_acc_std:.2f}%")
        print(f"Final Train Accuracy: {final_train_std:.2f}%")
        print(f"Final Test Accuracy: {final_test_std:.2f}%")
        
        # Find best overall model
        best_run_idx = np.argmax(best_test_accs)
        best_overall = all_results[best_run_idx]
        
        print("\n" + "="*60)
        print(f"BEST OVERALL MODEL (Run {best_run_idx+1})")
        print("="*60)
        print(f"Best Test Accuracy: {best_overall['best_test_acc']:.2f}%")
        print(f"95% Confidence Interval: [{best_overall['confidence_interval'][0]:.2f}%, {best_overall['confidence_interval'][1]:.2f}%]")
        print(f"From Epoch {best_overall['best_epoch']} of Run {best_run_idx+1}")
        
        # Print common words in misclassified examples from best model
        best_test_pred = best_overall['best_test_pred']
        test_y = best_overall['test_y']
        test_x = best_overall['test_x']
        id_to_word = best_overall['id_to_word']
        
        if np.sum(best_test_pred != test_y) > 0:
            print("\nCommon Words in Misclassified Examples from Best Model:")
            
            # Get misclassified examples
            misclassified_indices = np.where(best_test_pred != test_y)[0]
            
            # Get words from misclassified examples (excluding padding)
            misclassified_words = []
            for idx in misclassified_indices:
                for word_idx in test_x[idx]:
                    if word_idx > 0:  # Skip padding tokens
                        misclassified_words.append(id_to_word.get(word_idx, f"UNK_{word_idx}"))
            
            # Count word frequencies
            word_counter = Counter(misclassified_words)
            
            # Print most common words
            print("Most common words in misclassified examples:")
            for word, count in word_counter.most_common(20):
                print(f"{word}: {count}")
        
        print("\n" + "="*60)
        print(f"Model Parameters:")
        print(f"  - Number of Clauses: {args.number_of_clauses}")
        print(f"  - T: {args.T}")
        print(f"  - s: {args.s}")
        print(f"  - Depth: {args.depth}")
        print(f"  - Number of state bits: {args.number_of_state_bits}")
        print("="*60)
        
    else:
        # Run a single experiment if num_runs is 1
        results = run_single_experiment(args)
        
        # Analyze common words in misclassified examples from the single run
        best_test_pred = results['best_test_pred']
        test_y = results['test_y']
        test_x = results['test_x']
        id_to_word = results['id_to_word']
        
        print(f"ANALYZING BEST MODEL (Epoch {results['best_epoch']}, Test Accuracy: {results['best_test_acc']:.2f}%)")
        
       
        # Analyze common words in misclassified examples
        if np.sum(best_test_pred != test_y) > 0:
            print("\nCommon Words in Misclassified Examples:")
            
            # Get misclassified examples
            misclassified_indices = np.where(best_test_pred != test_y)[0]
            
            # Get words from misclassified examples (excluding padding)
            misclassified_words = []
            for idx in misclassified_indices:
                for word_idx in test_x[idx]:
                    if word_idx > 0:  # Skip padding tokens
                        misclassified_words.append(id_to_word.get(word_idx, f"UNK_{word_idx}"))
            
            # Count word frequencies
            word_counter = Counter(misclassified_words)
            
            # Print most common words
            print("Most common words in misclassified examples:")
            for word, count in word_counter.most_common(20):
                print(f"{word}: {count}")
            
            # Separate by false positives and false negatives
            false_pos_indices = np.where((best_test_pred == 1) & (test_y == 0))[0]
            false_neg_indices = np.where((best_test_pred == 0) & (test_y == 1))[0]
            
            if len(false_pos_indices) > 0:
                false_pos_words = []
                for idx in false_pos_indices:
                    for word_idx in test_x[idx]:
                        if word_idx > 0:  # Skip padding tokens
                            false_pos_words.append(id_to_word.get(word_idx, f"UNK_{word_idx}"))
                
                fp_counter = Counter(false_pos_words)
                print("\nMost common words in FALSE POSITIVES (negative examples classified as positive):")
                for word, count in fp_counter.most_common(15):
                    print(f"{word}: {count}")
            
            if len(false_neg_indices) > 0:
                false_neg_words = []
                for idx in false_neg_indices:
                    for word_idx in test_x[idx]:
                        if word_idx > 0:  # Skip padding tokens
                            false_neg_words.append(id_to_word.get(word_idx, f"UNK_{word_idx}"))
                
                fn_counter = Counter(false_neg_words)
                print("\nMost common words in FALSE NEGATIVES (positive examples classified as negative):")
                for word, count in fn_counter.most_common(15):
                    print(f"{word}: {count}")

if __name__ == "__main__":
    main()