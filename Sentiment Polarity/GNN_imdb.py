import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GlobalAttention
from torch_geometric.data import Data, Batch, DataLoader
from keras.datasets import imdb
import argparse
from tqdm import tqdm
import time

def default_args(**kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--hidden_size", default=64, type=int)
    parser.add_argument("--num_layers", default=2, type=int)
    parser.add_argument("--dropout", default=0.2, type=float)
    parser.add_argument("--learning_rate", default=0.001, type=float)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--imdb_num_words", default=10000, type=int)
    parser.add_argument("--imdb_index_from", default=2, type=int)
    parser.add_argument("--num_runs", default=3, type=int)  # Added for multiple runs
    
    args = parser.parse_args()
    for key, value in kwargs.items():
        if key in args.__dict__:
            setattr(args, key, value)
    return args

class IMDBGraphDataset:
    def __init__(self, data_x, data_y, num_words, index_from):
        self.data_x = data_x
        self.data_y = data_y
        self.num_words = num_words
        self.index_from = index_from
        
    def _count_edges(self, sequence_length):
        """Count edges for node based on position"""
        edge_index = []
        edge_attr = []
        max_offset = 4  # Same as original
        
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
        
        return torch.tensor(edge_index).t(), torch.tensor(edge_attr)
    
    def _create_graph(self, review, label):
        """Create a single graph from a review"""
        # Node features (one-hot encoded word IDs)
        x = torch.zeros((len(review), self.num_words))
        for i, word_id in enumerate(review):
            if word_id >= self.index_from:
                idx = word_id - self.index_from
                if idx < self.num_words:
                    x[i, idx] = 1.0
        
        # Create edges
        edge_index, edge_attr = self._count_edges(len(review))
        
        # Create graph
        return Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=torch.tensor([label]),
            num_nodes=len(review)
        )
    
    def create_dataset(self):
        """Convert all reviews to graphs"""
        graphs = []
        for review, label in tqdm(zip(self.data_x, self.data_y), desc="Creating graphs"):
            graph = self._create_graph(review, label)
            graphs.append(graph)
        return graphs

class IMDBGraphNN(nn.Module):
    def __init__(self, num_words, hidden_size, num_layers, num_edge_types=8):
        super().__init__()
        
        self.num_layers = num_layers
        
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
        
        # Process edge features
        edge_attr = self.edge_embedding(edge_attr)
        
        # Graph convolutions
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=0.2, training=self.training)
        
        # Global pooling
        x = self.pool(x, batch)
        
        # Classification
        return F.log_softmax(self.classifier(x), dim=-1)

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    batch_losses = []
    
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        out = model(batch)
        loss = F.nll_loss(out, batch.y)
        
        # Track per-batch loss for standard deviation calculation
        batch_losses.append(loss.item())
        
        # Track per-batch accuracy
        pred = out.argmax(dim=1)
        correct += (pred == batch.y).sum().item()
        total += batch.y.size(0)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(loader)
    accuracy = correct / total
    
    # Calculate loss standard deviation per batch
    loss_std = np.std(batch_losses) if batch_losses else 0
    
    return avg_loss, accuracy, loss_std

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    batch_accuracies = []
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            pred = out.argmax(dim=1)
            
            # Calculate per-batch accuracy
            batch_correct = (pred == batch.y).sum().item()
            batch_total = batch.y.size(0)
            batch_accuracies.append(batch_correct / batch_total)
            
            correct += batch_correct
            total += batch_total
    
    avg_accuracy = correct / total
    
    # Calculate accuracy standard deviation per batch
    acc_std = np.std(batch_accuracies) if batch_accuracies else 0
    
    return avg_accuracy, acc_std

def run_experiment(args, train_loader, test_loader, device, run_id):
    # Initialize model
    model = IMDBGraphNN(
        num_words=args.imdb_num_words,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers
    ).to(device)
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Metrics tracking
    epoch_losses = []
    epoch_train_accs = []
    epoch_test_accs = []
    epoch_loss_stds = []
    epoch_train_acc_stds = []
    epoch_test_acc_stds = []
    
    # Training loop
    print(f"\nStarting training for run {run_id + 1}/{args.num_runs}...")
    best_acc = 0
    for epoch in range(args.epochs):
        start_time = time.time()
        
        # Train
        loss, train_acc, loss_std = train_epoch(model, train_loader, optimizer, device)
        
        # Evaluate
        test_acc, test_acc_std = evaluate(model, test_loader, device)
        _, train_acc_std = evaluate(model, train_loader, device)  # Get training accuracy std
        
        # Store metrics
        epoch_losses.append(loss)
        epoch_train_accs.append(train_acc)
        epoch_test_accs.append(test_acc)
        epoch_loss_stds.append(loss_std)
        epoch_train_acc_stds.append(train_acc_std)
        epoch_test_acc_stds.append(test_acc_std)
        
        epoch_time = time.time() - start_time
        
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print(f"Loss: {loss:.4f} ± {loss_std:.4f}")
        print(f"Train Accuracy: {train_acc*100:.2f}% ± {train_acc_std*100:.2f}%")
        print(f"Test Accuracy: {test_acc*100:.2f}% ± {test_acc_std*100:.2f}%")
        print(f"Time: {epoch_time:.2f}s")
        
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), f"best_model_run_{run_id+1}.pt")
    
    print(f"\nRun {run_id + 1} - Best Test Accuracy: {best_acc*100:.2f}%")
    
    return {
        'losses': epoch_losses,
        'train_accs': epoch_train_accs,
        'test_accs': epoch_test_accs,
        'loss_stds': epoch_loss_stds,
        'train_acc_stds': epoch_train_acc_stds,
        'test_acc_stds': epoch_test_acc_stds,
        'best_acc': best_acc
    }

def calculate_metrics_across_runs(all_runs_metrics):
    """Calculate mean and standard deviation across multiple runs"""
    # Last epoch metrics across runs
    final_losses = [run['losses'][-1] for run in all_runs_metrics]
    final_train_accs = [run['train_accs'][-1] for run in all_runs_metrics]
    final_test_accs = [run['test_accs'][-1] for run in all_runs_metrics]
    best_accs = [run['best_acc'] for run in all_runs_metrics]
    
    # Calculate means
    mean_final_loss = np.mean(final_losses)
    mean_final_train_acc = np.mean(final_train_accs)
    mean_final_test_acc = np.mean(final_test_accs)
    mean_best_acc = np.mean(best_accs)
    
    # Calculate standard deviations
    std_final_loss = np.std(final_losses)
    std_final_train_acc = np.std(final_train_accs)
    std_final_test_acc = np.std(final_test_accs)
    std_best_acc = np.std(best_accs)
    
    return {
        'mean_final_loss': mean_final_loss,
        'mean_final_train_acc': mean_final_train_acc,
        'mean_final_test_acc': mean_final_test_acc,
        'mean_best_acc': mean_best_acc,
        'std_final_loss': std_final_loss,
        'std_final_train_acc': std_final_train_acc, 
        'std_final_test_acc': std_final_test_acc,
        'std_best_acc': std_best_acc
    }

def main():
    args = default_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load IMDB dataset
    print("Loading IMDB dataset...")
    (train_x, train_y), (test_x, test_y) = imdb.load_data(
        num_words=args.imdb_num_words,
        index_from=args.imdb_index_from
    )
    
    # Create graph datasets
    print("\nCreating training graphs...")
    train_dataset = IMDBGraphDataset(train_x, train_y, args.imdb_num_words, args.imdb_index_from)
    train_graphs = train_dataset.create_dataset()
    
    print("\nCreating testing graphs...")
    test_dataset = IMDBGraphDataset(test_x, test_y, args.imdb_num_words, args.imdb_index_from)
    test_graphs = test_dataset.create_dataset()
    
    # Create data loaders
    train_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=args.batch_size)
    
    # Run multiple experiments
    all_runs_metrics = []
    
    for run_id in range(args.num_runs):
        run_metrics = run_experiment(args, train_loader, test_loader, device, run_id)
        all_runs_metrics.append(run_metrics)
    
    # Calculate cross-run statistics
    cross_run_metrics = calculate_metrics_across_runs(all_runs_metrics)
    
    # Print final results
    print("\n" + "="*50)
    print("FINAL RESULTS ACROSS ALL RUNS")
    print("="*50)
    print(f"Number of runs: {args.num_runs}")
    print(f"Final Loss: {cross_run_metrics['mean_final_loss']:.4f} ± {cross_run_metrics['std_final_loss']:.4f}")
    print(f"Final Train Accuracy: {cross_run_metrics['mean_final_train_acc']*100:.2f}% ± {cross_run_metrics['std_final_train_acc']*100:.2f}%")
    print(f"Final Test Accuracy: {cross_run_metrics['mean_final_test_acc']*100:.2f}% ± {cross_run_metrics['std_final_test_acc']*100:.2f}%")
    print(f"Best Test Accuracy: {cross_run_metrics['mean_best_acc']*100:.2f}% ± {cross_run_metrics['std_best_acc']*100:.2f}%")
    print("="*50)

if __name__ == "__main__":
    main()