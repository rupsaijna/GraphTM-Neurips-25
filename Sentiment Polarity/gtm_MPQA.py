import numpy as np
import argparse
import os
import re
import glob
import random
from GraphTsetlinMachine.graphs import Graphs
from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine
from time import time
from collections import Counter
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
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
    parser.add_argument("--max-included-literals", default=None, type=int)
    parser.add_argument("--min-length", default=2, type=int)
    parser.add_argument("--max-length", default=160, type=int)
    parser.add_argument("--max-vocab-size", default=10000, type=int)
    parser.add_argument("--mpqa-dir", default="data/mpqa", type=str)
    parser.add_argument("--num-runs", default=3, type=int)
    parser.add_argument("--seed", default=42, type=int)
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
    # For binary classification, we can estimate standard deviation as sqrt(p*(1-p)/n)
    # where p is the accuracy and n is the number of samples
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

def regex_tokenize(text):
    """
    Tokenize test using regex to preserve contractions and words with apostrophes
    while removing commas and handling punctuation 
    """
    # First, replace commas with spaces to ensure they're removed
    text = text.replace(',', ' ')
    
    # - Words with apostrophes like "don't", "it's", "o'clock" as single tokens
    # - Regular words
    # - Excludes most punctuation as standalone tokens
    pattern = r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+"
    return re.findall(pattern, text.lower())

def load_mpqa_dataset(args):
    """
    Load the MPQA dataset from .pos and .neg files.
    Create a vocabulary with all unique words
    """
    print("Loading MPQA dataset...")
    
    # Ensure nltk tokenizer is available
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    
    # Check if dataset path exists
    if not os.path.exists(args.mpqa_dir):
        raise FileNotFoundError(f"MPQA dataset directory not found at {args.mpqa_dir}")
    
    # Find positive and negative files
    pos_file = os.path.join(args.mpqa_dir, "mpqa.pos")
    neg_file = os.path.join(args.mpqa_dir, "mpqa.neg")
    
    if not os.path.exists(pos_file) or not os.path.exists(neg_file):
        raise FileNotFoundError(f"MPQA files not found: {pos_file} or {neg_file}")
    
    # Load positive examples
    pos_texts = []
    with open(pos_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if line:
                pos_texts.append(line)
    
    # Load negative examples
    neg_texts = []
    with open(neg_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if line:
                neg_texts.append(line)
    
    # Original counts - no balancing
    original_pos_count = len(pos_texts)
    original_neg_count = len(neg_texts)
    print(f"Dataset: {original_pos_count} positive examples and {original_neg_count} negative examples")
    
    # Combine and create labels
    all_texts = pos_texts + neg_texts
    all_labels = [1] * len(pos_texts) + [0] * len(neg_texts)
    
    # Tokenize texts with regex tokenizer to preserve contractions
    tokenized_texts = [regex_tokenize(text) for text in all_texts]
    
    # Create temporary pairs for shuffling
    data_pairs = list(zip(tokenized_texts, all_labels))
    random.shuffle(data_pairs)
    tokenized_texts, all_labels = zip(*data_pairs)
    
    # Create vocabulary with all unique words
    print("Creating word vocabulary...")
    unique_words = set()
    for text in tokenized_texts:
        unique_words.update(text)
    
    # Add placeholder word for empty cases
    unique_words.add("<PLACEHOLDER>")
    
    # Sort vocabulary for consistency
    vocab = sorted(list(unique_words))
    print(f"Word vocabulary size: {len(vocab)}")
    
    # Create word to index mapping
    word_to_idx = {word: i for i, word in enumerate(vocab)}
    
    # Convert texts to sequences using word indices
    sequences = []
    for text in tokenized_texts:
        # Only include words that are in our vocabulary
        sequence = [word_to_idx[word] for word in text if word in word_to_idx]
        if sequence:  # Only add non-empty sequences
            sequences.append(sequence)
        else:
            # If all words are filtered out, add a placeholder
            placeholder_idx = word_to_idx["<PLACEHOLDER>"]
            sequences.append([placeholder_idx])
    
    # Check for empty sequences
    empty_sequences = sum(1 for seq in sequences if not seq)
    if empty_sequences > 0:
        print(f"Warning: {empty_sequences} sequences are empty after filtering")
    
    # Convert labels to numpy array
    labels = np.array(all_labels)
    
    # Split into train and test sets
    train_x, test_x, train_y, test_y = train_test_split(
        sequences, labels, test_size=0.3, random_state=42, stratify=labels
    )
   
    # Length statistics after processing
    train_lengths = [len(seq) for seq in train_x]
    test_lengths = [len(seq) for seq in test_x]
    
    
    return (train_x, train_y), (test_x, test_y), word_to_idx, vocab

def count_possible_edges(node_id, sequence_length):
    """Count possible edges for a node based on position."""
    possible_edges = 0
    max_offset = 4  # Maximum distance to connect words
    
    for offset in range(1, max_offset + 1):
        if node_id + offset < sequence_length:  # Forward edges
            possible_edges += 1
        if node_id - offset >= 0:              # Backward edges
            possible_edges += 1
    return possible_edges

def create_symbols(vocab):
    """
    Create word-level symbols without the "W_" prefix
    
    Parameters:
    vocab (list): List of words
    
    Returns:
    list: Symbols for words (the words themselves)
    """
    # Just use the words themselves as symbols
    word_symbols = vocab
    
    print(f"Number of word symbols: {len(word_symbols)}")
    print(f"Total symbols created: {len(word_symbols)}")
    
    return word_symbols

def configure_graphs(graphs, data_x, vocab, args, label="train"):
    # Phase 1: Set number of nodes (without type categorization)
    for graph_id, text in enumerate(data_x):
        # Ensure at least one node
        text_length = max(1, len(text))
        graphs.set_number_of_graph_nodes(graph_id, text_length)
    
    graphs.prepare_node_configuration()
    
    # Phase 2: Add nodes with edge counts
    for graph_id, text in enumerate(data_x):
        if len(text) <= 0:
            # Special case: empty text, add placeholder node with no edges
            node_name = f"Node_0"
            graphs.add_graph_node(graph_id, node_name, 0)
        else:
            for node_id in range(len(text)):
                possible_edges = count_possible_edges(node_id, len(text))
                # Add the node without type property
                node_name = f"Node_{node_id}"
                graphs.add_graph_node(graph_id, node_name, possible_edges)
    
    graphs.prepare_edge_configuration()
    
    # Phase 3: Add edges and word properties (no custom edge labels)
    for graph_id, text in enumerate(data_x):
        if len(text) <= 1:
            # For single-token texts, don't add edges
            if len(text) == 1:
                word_idx = text[0]
                if word_idx < len(vocab):
                    word = vocab[word_idx]  # Just use the word itself as the symbol
                    graphs.add_graph_node_property(graph_id, "Node_0", word)
        else:
            for node_id in range(len(text)):
                # Add forward and backward edges with numeric offsets
                max_offset = min(4, len(text) - 1)  # Limit offset to text length
                
                for offset in range(1, max_offset + 1):
                    # Backward edges
                    if node_id - offset >= 0:
                        graphs.add_graph_node_edge(graph_id, 
                                                f"Node_{node_id}", 
                                                f"Node_{node_id - offset}", 
                                                f"{-offset}")  # Just use numbers as edge labels
                    # Forward edges
                    if node_id + offset < len(text):
                        graphs.add_graph_node_edge(graph_id, 
                                                f"Node_{node_id}", 
                                                f"Node_{node_id + offset}", 
                                                f"{offset}")  # Just use numbers as edge labels
                
                # Add word property
                word_idx = text[node_id]
                if word_idx < len(vocab):  # Ensure index is in vocabulary
                    word = vocab[word_idx]  # Just use the word itself as the symbol
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
    
    # Load MPQA dataset with word features (not bigrams)
    (train_x, train_y), (test_x, test_y), word_to_idx, vocab = load_mpqa_dataset(args)
    
    # Create symbols for words
    symbols = create_symbols(vocab)
    
    # Initialize and configure training graphs
    print("\nInitializing training graphs...")
    graphs_train = Graphs(
        len(train_x),
        symbols=symbols,
        hypervector_size=args.hypervector_size,
        hypervector_bits=args.hypervector_bits
    )
    configure_graphs(graphs_train, train_x, vocab, args, "training")
    
    # Initialize and configure testing graphs
    print("\nInitializing testing graphs...")
    graphs_test = Graphs(len(test_x), init_with=graphs_train)
    configure_graphs(graphs_test, test_x, vocab, args, "testing")
    
    # Initialize Tsetlin Machine
    tm = MultiClassGraphTsetlinMachine(
        args.number_of_clauses,
        args.T,
        args.s,
        number_of_state_bits=args.number_of_state_bits,
        depth=args.depth,
        message_size=args.message_size,
        message_bits=args.message_bits,
        max_included_literals=args.max_included_literals,
        grid=(16 * 13 * 4, 1, 1),
        block=(128, 1, 1)
    )
    
    # Training loop
    print("\nStarting training...")
    best_test_acc = 0.0
    best_epoch = 0
    
    # Arrays to store accuracy at each epoch
    train_accuracies = []
    test_accuracies = []
    
    for epoch in range(args.epochs):
        start_training = time()
        tm.fit(graphs_train, train_y, epochs=1, incremental=True)
        stop_training = time()
        
        start_testing = time()
        # Make predictions
        train_pred = tm.predict(graphs_train)
        test_pred = tm.predict(graphs_test)
        
        # Calculate accuracies
        train_acc = 100 * (train_pred == train_y).mean()
        test_acc = 100 * (test_pred == test_y).mean()
        
        # Store accuracies for later standard deviation calculation
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        
        # Track best model
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch + 1
        
        # Track misclassifications and their lengths
        train_misclassified = train_pred != train_y
        test_misclassified = test_pred != test_y
        
        # Get lengths of all texts
        train_lengths = np.array([len(text) for text in train_x])
        test_lengths = np.array([len(text) for text in test_x])
        
        # Calculate statistics
        train_incorrect = np.sum(train_misclassified)
        test_incorrect = np.sum(test_misclassified)
        
       
        stop_testing = time()
        
        # Print results for each epoch
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print(f"Train Accuracy: {train_acc:.2f}%, Test Accuracy: {test_acc:.2f}%")
        print(f"Best Test Accuracy: {best_test_acc:.2f}% (Epoch {best_epoch})")
        print(f"Training Time: {stop_training - start_training:.2f}s")
        print(f"Testing Time: {stop_testing - start_testing:.2f}s")
        
    # Calculate standard deviation for this run
    train_acc_std = np.std(train_accuracies)
    test_acc_std = np.std(test_accuracies)
    
    # Print final results with standard deviation for this run
    print("\n" + "="*60)
    print(f"RUN {run_id+1} FINAL RESULTS")
    print("="*60)
    print(f"Best Test Accuracy: {best_test_acc:.2f}% (Epoch {best_epoch})")
    print(f"Train Accuracy Standard Deviation: {train_acc_std:.2f}%")
    print(f"Test Accuracy Standard Deviation: {test_acc_std:.2f}%")
    
    # Return final metrics for aggregation
    return {
        'best_test_acc': best_test_acc,
        'best_epoch': best_epoch,
        'train_accuracies': train_accuracies,
        'test_accuracies': test_accuracies,
        'train_acc_std': train_acc_std,
        'test_acc_std': test_acc_std
    }

def main():
    args = default_args()
    # Set default random seed
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
            final_train_accs.append(results['train_accuracies'][-1])
            final_test_accs.append(results['test_accuracies'][-1])
        
        # Calculate statistics across runs
        best_acc_mean = np.mean(best_test_accs)
        best_acc_std = np.std(best_test_accs)
        
        final_train_mean = np.mean(final_train_accs)
        final_train_std = np.std(final_train_accs)
        
        final_test_mean = np.mean(final_test_accs)
        final_test_std = np.std(final_test_accs)
        
        # Calculate epoch-wise statistics if all runs had the same number of epochs
        epoch_train_means = []
        epoch_train_stds = []
        epoch_test_means = []
        epoch_test_stds = []
        
        for epoch in range(args.epochs):
            epoch_train_accs = [results['train_accuracies'][epoch] for results in all_results 
                              if epoch < len(results['train_accuracies'])]
            epoch_test_accs = [results['test_accuracies'][epoch] for results in all_results 
                             if epoch < len(results['test_accuracies'])]
            
            if epoch_train_accs and epoch_test_accs:
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
        
        for epoch in range(len(epoch_train_means)):
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
        
        print("\n" + "="*60)
        print(f"Model Parameters:")
        print(f"  - Number of Clauses: {args.number_of_clauses}")
        print(f"  - T: {args.T}")
        print(f"  - s: {args.s}")
        print(f"  - Depth: {args.depth}")
        print(f"  - Stopword Removal: {args.remove_stopwords}")
        print("="*60)
        
    else:
        # Run a single experiment if num_runs is 1
        run_single_experiment(args)

if __name__ == "__main__":
    main()