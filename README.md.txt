#Master README

paper titled [The Tsetlin Machine Goes Deep: Logical Learning and Reasoning With Graphs]

## Installation

### 1. Create a Virtual Environment (Recommended)

It is recommended to use a Python virtual environment to keep the dependencies isolated:

```bash
# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate
```

### 2. Install Dependencies

After activating the virtual environment, install the required Python packages using the provided **requirements.txt** file:

```bash
pip install -r requirements.txt
```

### 3. Comparing with Coalesced Tsetlin Machine

Comparison with Coalesced TM requires installation via Git.

```bash
pip install git+https://github.com/cair/PySparseCoalescedTsetlinMachineCUDA
```

# Experiment : Multivalue  Noisy XOR experiment 
## Location : MultivalueNoisyXOR

## Training

To train the models in this experiment, run this command:

```train
python train.py
```
## Results

To recreate the figure, run this command

```results
python show_results.py
```

# Experiment : Color image classification tasks on CIFAR-10 dataset
## Location : CIFAR10

## Training and Testing

To run this experiment with GraphTM, run this command:

```train
python CIFAR10GraphTM.py
```

To run this experiment with CoalescedTM, run this command:

```train
python CIFAR10CoTM.py
```


# Experiment : Recommendation Systems
## Location : RecomendationSys

To run all experiments comparatively, run :

```train
python recommender_main.py
```

To run Graph Neural Network experiment with noise=0.1, run :

```train
python graph_nn.py --dataset_noise_ratio 0.1 --exp_id 1
```

To run Graph Tsetlin Machine experiment with noise=0.1, run :

```train
python graph_tm.py  --dataset_noise_ratio 0.1 --exp_id 1
```



# Experiment : Disconnected nodes on MNIST
## Location : MNIST

- Steps to train the GraphTM model:
    1. Run `python MNIST/gen_graphs.py` to generate and save the graphs from the MNIST dataset.
    2. Run `python MNIST/train_graphtm.py` to train and save the GraphTM model on the MNIST dataset.

- To plot the clauses,
    1. First make sure that the graphs and the trained model are saved.
    2. Run `python MNIST/print_clauses.py` to plot the clauses.

- To train the Convolutional CoTM model,
    1. Run `python MNIST/train_cotm.py` to train the Convolutional CoTM model on the MNIST dataset.


# Experiment : Disconnected nodes on  FMNIST
## Location : FMNIST

- Steps to train the GraphTM model:
    1. Run `python FMNIST/gen_graphs.py` to generate and save the graphs from the FMNIST dataset.
    2. Run `python FMNIST/train_graphtm.py` to train and save the GraphTM model on the FMNIST dataset.

- To plot the clauses,
    1. First make sure that the graphs and the trained model are saved.
    2. Run `python FMNIST/print_clauses.py` to plot the clauses.

- To train the Convolutional CoTM model,
    1. Run `python FMNIST/train_cotm.py` to train the Convolutional CoTM model on the FMNIST dataset.

# Experiment : Tracking Action Coreference on Tangram Dataset
## Location : Action Coreference

```train and test
python track_action_coreference.py
```

# Experiment : DNA Sequence classification
## Location : DNASequence

- Extract Data

```
mkdir -p data
unzip -o Sequence_Dataset.csv.zip -d data
```

- Train and test
python dnaseq_main.py

