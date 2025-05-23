import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GlobalAttention
from torch_geometric.data import Data, Batch, DataLoader
import argparse
from tqdm import tqdm
import time
import os
import pandas as pd
import re
import random
from sklearn.model_selection import train_test_split
from datasets import load_dataset

def default_args(**kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="mpqa", type=str, choices=["mpqa", "yelp"])
    # Define file directories within the code
    parser.add_argument("--data_dir", default="data/mpqa", type=str)
    parser.add_argument("--output_dir", default="./output", type=str)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--hidden_size", default=64, type=int)
    parser.add_argument("--num_layers", default=2, type=int)
    parser.add_argument("--dropout", default=0.2, type=float)
    parser.add_argument("--learning_rate", default=0.001, type=float)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--num_words", default=10000, type=int)
    parser.add_argument("--max_seq_length", default=200, type=int)
    parser.add_argument("--yelp_train_size", default=50000, type=int)
    parser.add_argument("--yelp_test_size", default=20000, type=int)
    parser.add_argument("--install_dependencies", action="store_true")
    parser.add_argument("--calculate_std", default=False, type=bool)
    parser.add_argument("--num_runs", default=5, type=int)
    
    args = parser.parse_args()
    for key, value in kwargs.items():
        if key in args.__dict__:
            setattr(args, key, value)
    return args

def set_seed(seed):
    """Set all random seeds to ensure reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Add this line to ensure reproducible behavior on CUDA side
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_mpqa_dataset(pos_file, neg_file):
    """Load MPQA dataset from positive and negative files"""
    texts = []
    labels = []
    
    # Load positive examples
    with open(pos_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            texts.append(line.strip())
            labels.append(1)  # Positive class
    
    # Load negative examples
    with open(neg_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            texts.append(line.strip())
            labels.append(0)  # Negative class
    
    # Shuffle the data
    combined = list(zip(texts, labels))
    random.shuffle(combined)
    texts, labels = zip(*combined)
    
    # Split into train and test sets (80% train, 20% test)
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )
    
    return (train_texts, train_labels), (test_texts, test_labels)

def load_yelp_dataset(train_size=50000, test_size=20000):
    """Load Yelp dataset from Hugging Face datasets and limit to specified sizes"""
  
    
    print("Loading Yelp dataset from Hugging Face...")
    # Load the Yelp dataset from Hugging Face
    dataset = load_dataset("yelp_review_full")
    
    # Extract train and test data
    train_data = dataset["train"]
    test_data = dataset["test"]
    
    # Convert ratings to binary sentiment (1-3 stars: negative, 4-5 stars: positive)
    train_texts = train_data["text"]
    train_labels = [1 if rating >= 4 else 0 for rating in train_data["label"]]
    
    test_texts = test_data["text"]
    test_labels = [1 if rating >= 4 else 0 for rating in test_data["label"]]
    
    # Create a balanced dataset with equal positive and negative samples
    pos_train_indices = [i for i, label in enumerate(train_labels) if label == 1]
    neg_train_indices = [i for i, label in enumerate(train_labels) if label == 0]
    
    pos_test_indices = [i for i, label in enumerate(test_labels) if label == 1]
    neg_test_indices = [i for i, label in enumerate(test_labels) if label == 0]
    
    # Determine how many samples of each class to take
    pos_train_size = min(train_size // 2, len(pos_train_indices))
    neg_train_size = min(train_size // 2, len(neg_train_indices))
    pos_test_size = min(test_size // 2, len(pos_test_indices))
    neg_test_size = min(test_size // 2, len(neg_test_indices))
    
    # Sample indices
    selected_pos_train = random.sample(pos_train_indices, pos_train_size)
    selected_neg_train = random.sample(neg_train_indices, neg_train_size)
    selected_pos_test = random.sample(pos_test_indices, pos_test_size)
    selected_neg_test = random.sample(neg_test_indices, neg_test_size)
    
    # Combine indices
    train_indices = selected_pos_train + selected_neg_train
    test_indices = selected_pos_test + selected_neg_test
    
    # Get the data
    sampled_train_texts = [train_texts[i] for i in train_indices]
    sampled_train_labels = [train_labels[i] for i in train_indices]
    sampled_test_texts = [test_texts[i] for i in test_indices]
    sampled_test_labels = [test_labels[i] for i in test_indices]
    
    print(f"Sampled {len(sampled_train_texts)} training examples and {len(sampled_test_texts)} test examples")
    print(f"  Positive train samples: {sum(sampled_train_labels)}")
    print(f"  Negative train samples: {len(sampled_train_labels) - sum(sampled_train_labels)}")
    print(f"  Positive test samples: {sum(sampled_test_labels)}")
    print(f"  Negative test samples: {len(sampled_test_labels) - sum(sampled_test_labels)}")
    
    return (sampled_train_texts, sampled_train_labels), (sampled_test_texts, sampled_test_labels)

def tokenize_and_build_vocab(texts, max_words=10000, max_seq_length=200):
    """Tokenize texts and build vocabulary"""
    # Simple tokenization function (can be improved with NLTK or spaCy)
    def tokenize(text):
        # Convert to lowercase and split by non-alphanumeric characters
        return re.findall(r'\w+', text.lower())
    
    # Tokenize all texts
    tokenized_texts = [tokenize(text) for text in texts]
    
    # Build vocabulary
    word_counts = {}
    for tokens in tokenized_texts:
        for token in tokens:
            if token in word_counts:
                word_counts[token] += 1
            else:
                word_counts[token] = 1
    
    # Sort words by frequency
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Create word-to-index mapping (reserve 0 for padding, 1 for unknown)
    word_to_idx = {'<PAD>': 0, '<UNK>': 1}
    for word, _ in sorted_words[:max_words - 2]:  # -2 for PAD and UNK
        word_to_idx[word] = len(word_to_idx)
    
    # Convert texts to sequences of indices
    sequences = []
    for tokens in tokenized_texts:
        seq = []
        for token in tokens[:max_seq_length]:
            if token in word_to_idx:
                seq.append(word_to_idx[token])
            else:
                seq.append(word_to_idx['<UNK>'])
        # Pad sequence if needed
        if len(seq) < max_seq_length:
            seq = seq + [word_to_idx['<PAD>']] * (max_seq_length - len(seq))
        sequences.append(seq)
    
    return sequences, word_to_idx

class TextGraphDataset:
    def __init__(self, data_x, data_y, num_words, index_from=2):
        self.data_x = data_x
        self.data_y = data_y
        self.num_words = num_words
        self.index_from = index_from
        
    def _count_edges(self, sequence_length):
        """Count edges for node based on position"""
        edge_index = []
        edge_attr = []
        max_offset = 4  # Same as original
        
        # If sequence is too short, add at least a self-loop
        if sequence_length <= 1:
            return torch.tensor([[0], [0]], dtype=torch.long), torch.tensor([0], dtype=torch.long)
        
        for node_id in range(sequence_length):
            for offset in range(1, max_offset + 1):
                # Forward edges
                if node_id + offset < sequence_length:
                    edge_index.append([node_id, node_id + offset])
                    edge_attr.append(offset - 1)  # 0-3 for forward
                # Backward edges
                if node_id - offset >= 0:
                    edge_index.append([node_id, node_id - offset])
                    edge_attr.append(offset + 3)  # 4-7 for backward
        
        # Check if we created any edges
        if len(edge_index) == 0:
            # Add a self-loop as fallback
            return torch.tensor([[0], [0]], dtype=torch.long), torch.tensor([0], dtype=torch.long)
            
        return torch.tensor(edge_index, dtype=torch.long).t(), torch.tensor(edge_attr, dtype=torch.long)
    
    def _create_graph(self, review, label):
        """Create a single graph from a review"""
        # Remove padding tokens
        review = [word_id for word_id in review if word_id != 0]
        
        # Handle empty reviews or reviews with only padding
        if len(review) == 0:
            # Create a minimal valid graph with at least one self-loop
            return Data(
                x=torch.zeros((1, self.num_words)),
                edge_index=torch.tensor([[0], [0]], dtype=torch.long),  # Self-loop
                edge_attr=torch.tensor([0], dtype=torch.long),
                y=torch.tensor([label], dtype=torch.long),
                num_nodes=1
            )
        
        # Node features (one-hot encoded word IDs)
        x = torch.zeros((len(review), self.num_words))
        for i, word_id in enumerate(review):
            if word_id >= self.index_from:
                idx = word_id - self.index_from
                if idx < self.num_words:
                    x[i, idx] = 1.0
        
        # Create edges
        edge_index, edge_attr = self._count_edges(len(review))
        
        # If no edges were created, add a self-loop to avoid errors
        if edge_index.size(1) == 0:
            edge_index = torch.tensor([[0], [0]], dtype=torch.long)
            edge_attr = torch.tensor([0], dtype=torch.long)
        
        # Create graph
        return Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=torch.tensor([label], dtype=torch.long),
            num_nodes=len(review)
        )
    
    def create_dataset(self):
        """Convert all reviews to graphs"""
        graphs = []
        for review, label in tqdm(zip(self.data_x, self.data_y), desc="Creating graphs"):
            graph = self._create_graph(review, label)
            if graph.num_nodes > 0:  # Include the graph even if it's minimal
                graphs.append(graph)
        return graphs

class TextGraphNN(nn.Module):
    def __init__(self, num_words, hidden_size, num_layers, num_edge_types=8, dropout=0.2):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout_rate = dropout
        
        # Edge embedding
        self.edge_embedding = nn.Embedding(num_edge_types, hidden_size)
        
        # Graph layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # First layer
        self.convs.append(GATConv(num_words, hidden_size))
        self.batch_norms.append(nn.BatchNorm1d(hidden_size))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(GATConv(hidden_size, hidden_size))
            self.batch_norms.append(nn.BatchNorm1d(hidden_size))
        
        # Global pooling
        gate_nn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        self.pool = GlobalAttention(gate_nn)
        
        # Output layer
        self.classifier = nn.Linear(hidden_size, 2)
    
    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # Ensure edge_index and edge_attr are long tensors (integer type)
        edge_index = edge_index.long()
        edge_attr = edge_attr.long()
        
        # Handle empty edge_index tensor
        if edge_index.numel() == 0:
            # If no edges, create a dummy output and return
            dummy_output = torch.zeros((batch.max().item() + 1, 2), device=x.device)
            return F.log_softmax(dummy_output, dim=-1)
        
        # Process edge features
        edge_attr = self.edge_embedding(edge_attr)
        
        # Graph convolutions
        for i in range(self.num_layers):
            # Check if edge_index is valid
            if edge_index.size(1) == 0:
                # Skip convolution if there are no edges
                continue
                
            x = self.convs[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
        
        # Global pooling
        x = self.pool(x, batch)
        
        # Classification
        return F.log_softmax(self.classifier(x), dim=-1)

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        out = model(batch)
        loss = F.nll_loss(out, batch.y)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(loader)

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            pred = out.argmax(dim=1)
            correct += (pred == batch.y).sum().item()
            total += batch.y.size(0)
    
    return correct / total

def run_training(train_loader, test_loader, args, device, run_id=None):
    """
    Run a full training cycle and return metrics.
    
    Parameters:
    -----------
    train_loader : DataLoader
        DataLoader for training data.
    test_loader : DataLoader
        DataLoader for test data.
    args : Namespace
        Command line arguments.
    device : torch.device
        Device to run on.
    run_id : int or None
        ID of the current run (for multi-run experiments).
        
    Returns:
    --------
    epoch_accuracies : list
        List of test accuracies for each epoch.
    best_acc : float
        Best test accuracy achieved.
    """
    # Initialize model
    model = TextGraphNN(
        num_words=args.num_words,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Store epoch accuracies for each run
    epoch_accuracies = []
    
    # Training loop
    run_str = f" (Run {run_id+1}/{args.num_runs})" if run_id is not None else ""
    print(f"\nStarting training{run_str}...")
    best_acc = 0
    
    for epoch in range(args.epochs):
        start_time = time.time()
        
        # Train
        loss = train_epoch(model, train_loader, optimizer, device)
        
        # Evaluate
        train_acc = evaluate(model, train_loader, device)
        test_acc = evaluate(model, test_loader, device)
        
        epoch_accuracies.append(test_acc)
        
        epoch_time = time.time() - start_time
        
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print(f"Loss: {loss:.4f}")
        print(f"Train Accuracy: {train_acc*100:.2f}%")
        print(f"Test Accuracy: {test_acc*100:.2f}%")
        print(f"Time: {epoch_time:.2f}s")
        
        if test_acc > best_acc:
            best_acc = test_acc
            if args.output_dir:
                # Only save model when not doing multiple runs
                if not args.calculate_std or run_id is None:
                    model_path = os.path.join(args.output_dir, f"best_model_{args.dataset}.pt")
                    torch.save(model.state_dict(), model_path)
                    print(f"Saved best model to {model_path}")
    
    print(f"\nBest Test Accuracy: {best_acc*100:.2f}%")
    return epoch_accuracies, best_acc

def run_multiple_experiments(train_graphs, test_graphs, args, device):
    """
    Run multiple training experiments to calculate standard deviation.
    
    Parameters:
    -----------
    train_graphs : list
        List of training graph data objects.
    test_graphs : list
        List of test graph data objects.
    args : Namespace
        Command line arguments.
    device : torch.device
        Device to run on.
        
    Returns:
    --------
    mean_accuracy : float
        Mean of best accuracies across all runs.
    std_accuracy : float
        Standard deviation of best accuracies across all runs.
    """
    all_best_accuracies = []
    all_epoch_accuracies = []
    
    print(f"\n{'='*80}")
    print(f"Running {args.num_runs} experiments to calculate standard deviation")
    print(f"{'='*80}\n")
    
    for run in range(args.num_runs):
        # Set different seeds for each run
        set_seed(42 + run)
        
        # Create data loaders for each run with the same data but different shuffling
        train_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_graphs, batch_size=args.batch_size)
        
        # Run full training cycle
        epoch_accuracies, best_acc = run_training(train_loader, test_loader, args, device, run)
        
        all_best_accuracies.append(best_acc)
        all_epoch_accuracies.append(epoch_accuracies)
    
    # Calculate mean and standard deviation of best accuracies
    mean_accuracy = np.mean(all_best_accuracies)
    std_accuracy = np.std(all_best_accuracies)
    
    print(f"\n{'='*80}")
    print(f"Results across {args.num_runs} runs:")
    print(f"Mean of best accuracies: {mean_accuracy*100:.2f}%")
    print(f"Standard deviation of best accuracies: {std_accuracy*100:.2f}%")
    print(f"All best accuracies: {[f'{acc*100:.2f}%' for acc in all_best_accuracies]}")
    print(f"{'='*80}")
    
    # Calculate per-epoch statistics
    if args.epochs > 1:
        epoch_means = np.mean(all_epoch_accuracies, axis=0)
        epoch_stds = np.std(all_epoch_accuracies, axis=0)
        
        print("\nEpoch-wise statistics:")
        print(f"{'Epoch':<10}{'Mean Acc (%)':<15}{'Std Dev (%)':<15}")
        print(f"{'-'*40}")
        for epoch in range(args.epochs):
            print(f"{epoch+1:<10}{epoch_means[epoch]*100:<15.2f}{epoch_stds[epoch]*100:<15.2f}")
    
    return mean_accuracy, std_accuracy, all_epoch_accuracies

def main():
    args = default_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Set the initial seed for reproducibility
    set_seed(42)
    
    # Install required dependencies if needed
    if args.install_dependencies:
        import subprocess
        print("Installing required dependencies...")
        subprocess.check_call(["pip", "install", "datasets", "huggingface_hub"])
    
    # Create directories if they don't exist
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Define file paths for MPQA dataset
    mpqa_pos_file = os.path.join(args.data_dir, "mpqa.pos")
    mpqa_neg_file = os.path.join(args.data_dir, "mpqa.neg")
    
    # Load dataset based on argument
    if args.dataset == "mpqa":
        print("Loading MPQA dataset...")
        print(f"MPQA positive file: {mpqa_pos_file}")
        print(f"MPQA negative file: {mpqa_neg_file}")
        (train_texts, train_labels), (test_texts, test_labels) = load_mpqa_dataset(
            mpqa_pos_file, mpqa_neg_file
        )
    else:  # yelp
        print("Loading Yelp dataset from Hugging Face...")
        (train_texts, train_labels), (test_texts, test_labels) = load_yelp_dataset(
            args.yelp_train_size, args.yelp_test_size
        )
    
    # Tokenize and convert to sequences
    print("Tokenizing and building vocabulary...")
    all_texts = train_texts + test_texts
    all_sequences, word_to_idx = tokenize_and_build_vocab(
        all_texts, args.num_words, args.max_seq_length
    )
    
    train_sequences = all_sequences[:len(train_texts)]
    test_sequences = all_sequences[len(train_texts):]
    
    # Create graph datasets
    print("\nCreating training graphs...")
    train_dataset = TextGraphDataset(train_sequences, train_labels, args.num_words)
    train_graphs = train_dataset.create_dataset()
    
    print("\nCreating testing graphs...")
    test_dataset = TextGraphDataset(test_sequences, test_labels, args.num_words)
    test_graphs = test_dataset.create_dataset()
    
    # Decide whether to run multiple experiments or a single run
    if args.calculate_std and args.num_runs > 1:
        mean_accuracy, std_accuracy, all_epoch_accuracies = run_multiple_experiments(
            train_graphs, test_graphs, args, device
        )
    else:
        # Regular single run
        train_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_graphs, batch_size=args.batch_size)
        
        epoch_accuracies, best_acc = run_training(train_loader, test_loader, args, device)

if __name__ == "__main__":
    main()