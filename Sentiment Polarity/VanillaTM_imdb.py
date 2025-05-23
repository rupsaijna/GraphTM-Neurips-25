import argparse
from tmu.models.classification.vanilla_classifier import TMClassifier
from tmu.tools import BenchmarkTimer
import numpy as np
from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import CountVectorizer

def prepare_imdb_dataset(num_words=10000):

    # Load the IMDB dataset from TensorFlow
    print("Loading IMDB dataset from TensorFlow...")
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=num_words)
    
    # Get the word index mapping from imdb
    word_index = imdb.get_word_index()
    
    # Create reverse mapping to convert indices back to words
    reverse_word_index = {i: word for word, i in word_index.items()}
    
    # Function to convert index sequence to text
    def sequence_to_text(sequence):
        # 0, 1, and 2 are reserved indices in the IMDB dataset
        return ' '.join([reverse_word_index.get(i - 3, '?') for i in sequence if i > 2])
    
    # Convert sequences to text
    print("Converting sequences back to text...")
    train_texts = [sequence_to_text(seq) for seq in x_train]
    test_texts = [sequence_to_text(seq) for seq in x_test]
    
    # Use CountVectorizer to create binary features
    print("Creating binary features with CountVectorizer...")
    vectorizer = CountVectorizer(binary=True, max_features=num_words)
    
    # Fit on training data and transform both training and test data
    X_train = vectorizer.fit_transform(train_texts).astype(np.uint32).toarray()
    X_test = vectorizer.transform(test_texts).astype(np.uint32).toarray()
    
    # Convert labels to numpy arrays with the required uint32 data type
    Y_train = np.array(y_train, dtype=np.uint32)
    Y_test = np.array(y_test, dtype=np.uint32)
    
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

    all_accuracies = []
    all_epoch_accuracies = []
    
    print(f"\n{'='*80}")
    print(f"Running {num_runs} experiments to calculate standard deviation")
    print(f"{'='*80}\n")
    
    # Load dataset - only need to do this once for multiple runs
    print("Loading dataset once for all runs...")
    X_train, X_test, Y_train, Y_test = prepare_imdb_dataset(num_words=args.num_words)
    
    for run in range(num_runs):
        print(f"\nRun {run+1}/{num_runs}")
        print(f"{'-'*50}")
        
        # Different random seeds for each run
        np.random.seed(run)
        tf.random.set_seed(run)  # Set TensorFlow seed as well
        
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
        X_train, X_test, Y_train, Y_test = prepare_imdb_dataset(num_words=args.num_words)
        accuracy, total_time, epoch_accuracies = train_tmu_classifier(X_train, X_test, Y_train, Y_test, args)
        return accuracy, total_time, epoch_accuracies

def default_args(**kwargs):
    parser = argparse.ArgumentParser()
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
    
   
   
# file.py --epochs 5 --calculate_std True --num_runs 5