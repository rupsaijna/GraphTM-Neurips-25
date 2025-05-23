
import argparse
import os
import numpy as np
from tmu.models.classification.vanilla_classifier import TMClassifier
from tmu.tools import BenchmarkTimer
from sklearn.feature_extraction.text import CountVectorizer
from datasets import load_dataset

def prepare_yelp_dataset(num_words=10000, train_size=50000, test_size=20000):
    """
    Prepare the Yelp dataset from Hugging Face for TMU classification.
    
    Parameters:
    -----------
    num_words : int
        Maximum number of words to include in the vocabulary.
    train_size : int
        Number of samples to use for training (default: 50000).
    test_size : int
        Number of samples to use for testing (default: 20000).
        
    Returns:
    --------
    X_train, X_test, Y_train, Y_test : arrays
        Training and testing data in binary format for TMU.
    """
    print("Loading Yelp dataset from Hugging Face...")
    # Load the Yelp dataset - polarity/sentiment dataset
    dataset = load_dataset("yelp_review_full")
    
    # The Yelp dataset has ratings from 1 to 5, convert it to binary sentiment
    # (1-2: negative, 4-5: positive, ignoring 3 as neutral)
    train_texts = []
    train_labels = []
    train_count = {'neg': 0, 'pos': 0}
    max_per_class = train_size // 2  # Equal distribution of positive and negative examples
    
    for item in dataset["train"]:
        if item["label"] in [0, 1] and train_count['neg'] < max_per_class:  # 1 and 2 stars (negative)
            train_texts.append(item["text"])
            train_labels.append(0)  # negative
            train_count['neg'] += 1
        elif item["label"] in [3, 4] and train_count['pos'] < max_per_class:  # 4 and 5 stars (positive)
            train_texts.append(item["text"])
            train_labels.append(1)  # positive
            train_count['pos'] += 1
            
        # Stop if we've collected enough samples
        if train_count['neg'] >= max_per_class and train_count['pos'] >= max_per_class:
            break
    
    test_texts = []
    test_labels = []
    test_count = {'neg': 0, 'pos': 0}
    max_test_per_class = test_size // 2  # Equal distribution
    
    for item in dataset["test"]:
        if item["label"] in [0, 1] and test_count['neg'] < max_test_per_class:  # 1 and 2 stars (negative)
            test_texts.append(item["text"])
            test_labels.append(0)  # negative
            test_count['neg'] += 1
        elif item["label"] in [3, 4] and test_count['pos'] < max_test_per_class:  # 4 and 5 stars (positive)
            test_texts.append(item["text"])
            test_labels.append(1)  # positive
            test_count['pos'] += 1
            
        # Stop if we've collected enough samples
        if test_count['neg'] >= max_test_per_class and test_count['pos'] >= max_test_per_class:
            break
    
    print(f"Collected {len(train_texts)} training samples and {len(test_texts)} test samples")
    
    # Use CountVectorizer to create binary features
    print("Creating binary features with CountVectorizer...")
    vectorizer = CountVectorizer(binary=True, max_features=num_words)
    
    # Fit on training data and transform both training and test data
    X_train = vectorizer.fit_transform(train_texts).astype(np.uint32).toarray()
    X_test = vectorizer.transform(test_texts).astype(np.uint32).toarray()
    
    # Convert labels to numpy arrays with the required uint32 data type
    Y_train = np.array(train_labels, dtype=np.uint32)
    Y_test = np.array(test_labels, dtype=np.uint32)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")
    
    return X_train, X_test, Y_train, Y_test

def prepare_mpqa_dataset(pos_file, neg_file, test_split=0.2, num_words=10000):
    """
    Prepare the MPQA dataset from local files for TMU classification.
    
    Parameters:
    -----------
    pos_file : str
        Path to the positive samples file.
    neg_file : str
        Path to the negative samples file.
    test_split : float
        Proportion of data to use for testing.
    num_words : int
        Maximum number of words to include in the vocabulary.
        
    Returns:
    --------
    X_train, X_test, Y_train, Y_test : arrays
        Training and testing data in binary format for TMU.
    """
    print("Loading MPQA dataset from files...")
    
    # Read the positive and negative samples from files
    with open(pos_file, 'r', encoding='utf-8', errors='ignore') as f:
        positive_texts = [line.strip() for line in f if line.strip()]
    
    with open(neg_file, 'r', encoding='utf-8', errors='ignore') as f:
        negative_texts = [line.strip() for line in f if line.strip()]
    
    # Create labels
    positive_labels = [1] * len(positive_texts)
    negative_labels = [0] * len(negative_texts)
    
    # Combine texts and labels
    all_texts = positive_texts + negative_texts
    all_labels = positive_labels + negative_labels
    
    # Shuffle the data
    indices = np.random.permutation(len(all_texts))
    all_texts = [all_texts[i] for i in indices]
    all_labels = [all_labels[i] for i in indices]
    
    # Split the data into training and testing sets
    split_idx = int(len(all_texts) * (1 - test_split))
    train_texts = all_texts[:split_idx]
    train_labels = all_labels[:split_idx]
    test_texts = all_texts[split_idx:]
    test_labels = all_labels[split_idx:]
    
    # Use CountVectorizer to create binary features
    print("Creating binary features with CountVectorizer...")
    vectorizer = CountVectorizer(binary=True, max_features=num_words)
    
    # Fit on training data and transform both training and test data
    X_train = vectorizer.fit_transform(train_texts).astype(np.uint32).toarray()
    X_test = vectorizer.transform(test_texts).astype(np.uint32).toarray()
    
    # Convert labels to numpy arrays with the required uint32 data type
    Y_train = np.array(train_labels, dtype=np.uint32)
    Y_test = np.array(test_labels, dtype=np.uint32)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")
    
    return X_train, X_test, Y_train, Y_test

def train_tmu_classifier(X_train, X_test, Y_train, Y_test, args):
   
    tm = TMClassifier(
        number_of_clauses=args.num_clauses,
        T=args.T,
        s=args.s,
        max_included_literals=args.max_included_literals,
        platform=args.platform,
        weighted_clauses=args.weighted_clauses,
    )
    
    # Store accuracy for each epoch
    epoch_accuracies = []
    
    benchmark_total = BenchmarkTimer(logger=None, text="Total Time")
    with benchmark_total:
        for epoch in range(args.epochs):
            benchmark1 = BenchmarkTimer(logger=None, text="Training Time")
            with benchmark1:
                tm.fit(X_train, Y_train)
            train_time = benchmark1.elapsed()
            
            benchmark2 = BenchmarkTimer(logger=None, text="Testing Time")
            with benchmark2:
                accuracy = 100 * (tm.predict(X_test) == Y_test).mean()
            test_time = benchmark2.elapsed()
            
            epoch_accuracies.append(accuracy)
            
            print(f"Epoch {epoch+1}/{args.epochs}, Accuracy: {accuracy:.2f}%, "
                  f"Training Time: {train_time:.2f}s, Testing Time: {test_time:.2f}s")
    
    total_time = benchmark_total.elapsed()
    final_accuracy = epoch_accuracies[-1]
    print(f"Final accuracy: {final_accuracy:.2f}%")
    print(f"Total time: {total_time:.2f}s")
    
    return final_accuracy, total_time, epoch_accuracies

def run_multiple_experiments(args, num_runs=5):
    """
    Run multiple experiments to calculate mean and standard deviation of accuracy.
    
    Parameters:
    -----------
    args : Namespace
        Command line arguments.
    num_runs : int
        Number of experimental runs.
        
    Returns:
    --------
    mean_accuracy : float
        Mean accuracy across all runs.
    std_accuracy : float
        Standard deviation of accuracy across all runs.
    """
    all_accuracies = []
    all_epoch_accuracies = []
    
    print(f"\n{'='*80}")
    print(f"Running {num_runs} experiments to calculate standard deviation")
    print(f"{'='*80}\n")
    
    for run in range(num_runs):
        print(f"\nRun {run+1}/{num_runs}")
        print(f"{'-'*50}")
        
        # Load dataset - only need to do this once for multiple runs
        if run == 0:
            if args.dataset == "yelp":
                X_train, X_test, Y_train, Y_test = prepare_yelp_dataset(
                    num_words=args.num_words,
                    train_size=args.train_size,
                    test_size=args.test_size
                )
            elif args.dataset == "mpqa":
                X_train, X_test, Y_train, Y_test = prepare_mpqa_dataset(
                    pos_file=args.mpqa_pos_file,
                    neg_file=args.mpqa_neg_file,
                    test_split=args.test_split,
                    num_words=args.num_words
                )
            else:
                raise ValueError(f"Unknown dataset: {args.dataset}")
        
        # Different random seeds for each run
        np.random.seed(run)
        
        accuracy, _, epoch_accuracies = train_tmu_classifier(X_train, X_test, Y_train, Y_test, args)
        all_accuracies.append(accuracy)
        all_epoch_accuracies.append(epoch_accuracies)
    
    # Calculate mean and standard deviation
    mean_accuracy = np.mean(all_accuracies)
    std_accuracy = np.std(all_accuracies)
    
    print(f"\n{'='*80}")
    print(f"Results across {num_runs} runs:")
    print(f"Mean accuracy: {mean_accuracy:.2f}%")
    print(f"Standard deviation: {std_accuracy:.2f}%")
    print(f"All accuracies: {all_accuracies}")
    print(f"{'='*80}")
    
    # Calculate per-epoch statistics
    if args.epochs > 1:
        epoch_means = np.mean(all_epoch_accuracies, axis=0)
        epoch_stds = np.std(all_epoch_accuracies, axis=0)
        
        print("\nEpoch-wise statistics:")
        print(f"{'Epoch':<10}{'Mean Acc (%)':<15}{'Std Dev (%)':<15}")
        print(f"{'-'*40}")
        for epoch in range(args.epochs):
            print(f"{epoch+1:<10}{epoch_means[epoch]:<15.2f}{epoch_stds[epoch]:<15.2f}")
    
    return mean_accuracy, std_accuracy, all_epoch_accuracies

def main(args):
    if args.calculate_std and args.num_runs > 1:
        mean_accuracy, std_accuracy, epoch_accuracies = run_multiple_experiments(args, args.num_runs)
        return mean_accuracy, std_accuracy, epoch_accuracies
    else:
        if args.dataset == "yelp":
            X_train, X_test, Y_train, Y_test = prepare_yelp_dataset(
                num_words=args.num_words,
                train_size=args.train_size,
                test_size=args.test_size
            )
        elif args.dataset == "mpqa":
            X_train, X_test, Y_train, Y_test = prepare_mpqa_dataset(
                pos_file=args.mpqa_pos_file,
                neg_file=args.mpqa_neg_file,
                test_split=args.test_split,
                num_words=args.num_words
            )
        else:
            raise ValueError(f"Unknown dataset: {args.dataset}")
        
        accuracy, total_time, epoch_accuracies = train_tmu_classifier(X_train, X_test, Y_train, Y_test, args)
        return accuracy, total_time, epoch_accuracies

def default_args(**kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["yelp", "mpqa"], required=True,help="Which dataset to use)")
    parser.add_argument("--train_size", default=50000, type=int,help="Number of samples to use for training Yelp ")
    parser.add_argument("--test_size", default=20000, type=int,help="samples to use for testing Yelp ")
    parser.add_argument("--mpqa_pos_file", default="data/mpqa.pos", type=str)
    parser.add_argument("--mpqa_neg_file", default="data/mpqa.neg", type=str)
    parser.add_argument("--test_split", default=0.2, type=float, help="testing MPQA only")
    parser.add_argument("--num_clauses", default=10000, type=int)
    parser.add_argument("--T", default=100000, type=int)
    parser.add_argument("--s", default=15.0, type=float)
    parser.add_argument("--max_included_literals", default=32, type=int)
    parser.add_argument("--platform", default="CPU_sparse", type=str, choices=["CPU", "CPU_sparse", "CUDA"])
    parser.add_argument("--weighted_clauses", default=True, type=bool)
    parser.add_argument("--epochs", default=40, type=int)
    parser.add_argument("--num_words", default=10000, type=int, help="Size of the vocabulary")
    parser.add_argument("--calculate_std", default=True, type=bool)
    parser.add_argument("--num_runs", default=3, type=int)
 
    args = parser.parse_args()
    for key, value in kwargs.items():
        if key in args.__dict__:
            setattr(args, key, value)
    return args

if __name__ == "__main__":
    main(default_args())
    

# example run
# file.py --dataset yelp --epochs 5 --calculate_std True --num_runs 5