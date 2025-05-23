#README

Supplemental Material for Paper titled [The Tsetlin Machine Goes Deep: Logical Learning and Reasoning With Graphs]

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
 
### 3. Install GraphTM

```bash
python ./setup.py sdist
pip3 install dist/GraphTsetlinMachine-0.3.3.tar.gz
```


### 4. Comparing with Standard Tsetlin Machine

Comparison with Standard TM requires installation via pip.

```bash
pip install tmu
```

### 5. Comparing with Coalesced Tsetlin Machine

Comparison with Coalesced TM requires installation via Git/pip.

```bash
pip install git+https://github.com/cair/PySparseCoalescedTsetlinMachineCUDA
```

#Experiments reported in the paper

## Experiment : Multivalue  Noisy XOR experiment 
### Location : Multivalue NoisyXOR


-To train the models in this experiment, run:

```train
python train.py
```
-To recreate the figure, run :

```results
python show_results.py
```

## Experiment : Color image classification tasks on CIFAR-10 dataset
### Location : CIFAR10


-For CIFAR10 experiment with GraphTM, run :

```train
python CIFAR10GraphTM.py
```

-For CIFAR10 experiment with CoalescedTM, run :

```train
python CIFAR10CoTM.py
```


## Experiment : Recommendation System
### Location : Recommendation System

-To execute all experiments comparatively, run :

```train
python recommender_main.py
```

-For Graph Neural Network experiment with noise=0.1, run :

```train
python graph_nn.py --dataset_noise_ratio 0.1 --exp_id 1
```

-For GraphTM experiment with noise=0.1, run :

```bash
python graph_tm.py  --dataset_noise_ratio 0.1 --exp_id 1
```



## Experiment : Disconnected nodes on MNIST
### Location : MNIST

- To train GraphTM model:
    
    - To generate and save graph from MNIST to use as input to GraphTM, run:
    ```bash
    python MNIST/gen_graphs.py
    ```
    
    - To train and save GraphTM Model, run:
    ```bash
    python MNIST/train_graphtm.py
    ```

- To plot the clauses,
        First make sure that the graphs and the trained model are saved.
        Then run:
    ```bash
        python MNIST/print_clauses.py
    ``` 

- To experiment with Convolutional CoTM model, run:
```bash
        python MNIST/train_cotm.py
``` 


## Experiment : Disconnected nodes on  FMNIST
### Location : FMNIST

- To train GraphTM model:
    
    - To generate and save graph from FMNIST to use as input to GraphTM, run
    ```bash
    python FMNIST/gen_graphs.py
    ```
    
    - To train and save GraphTM Model, run:
    ```bash
    python FMNIST/train_graphtm.py
    ```

- To plot the clauses,
        First make sure that the graphs and the trained model are saved.
        Then run:
    ```bash
        python FMNIST/print_clauses.py
    ``` 

- To experiment with Convolutional CoTM model, run:
```bash
        python FMNIST/train_cotm.py
``` 

## Experiment : Tracking Action Coreference on Tangram Dataset
### Location : Action Coreference

-To train and test, run :
```bash
python track_action_coreference.py
```

## Experiment : DNA Sequence classification
### Location : DNASequence

- Extract Data

```
mkdir -p data
unzip -o Sequence_Dataset.csv.zip -d data
```

- To train and test, run:
```
python dnaseq_main.py
```

## Experiment : Sentiment Polarity classification
### Location : Sentiment Polarity

-For experiment on IMDB/MPQA/Yelp with GraphTM, run :

```train
python gtm_imdb.py
python gtm_MPQA.py
python gtm_yelp.py
```

-For experiment on IMDB/MPQA/Yelp with GNN, run :

```train
python GNN_IMDB.py
python GNN_mpqa&Yelp.py
```

-For experiment on IMDB/MPQA/Yelp with Vanilla TM, run :

```train
python VanillaTM_imdb.py
python VanillaTM__mpqa&Yelp.py
```


------

## Implementation details using NoisyXOR

For Noisy XOR each node has one of two properties: **A** or **B**. If both of the graph's nodes have the same property, the graph is given the class label $Y=0$. Otherwise, it is given the class label $Y=1$.

The task of the Graph Tsetlin Machine is to assign the correct class label to each graph when the labels used for training are noisy.

### Initialization

Start by creating the training graphs using the _Graphs_ construct:
```bash
graphs_train = Graphs(
    10000,
    symbols = ['A', 'B'],
    hypervector_size = 32,
    hypervector_bits = 2
)
```
You initialize the graphs as follows:
- *Number of Graphs.* The first number sets how many graphs you are going to create. Here, you prepare for creating _10,000_ graphs.

- *Symbols.* Next, you find the symbols **A** and **B**. You use these symbols to assign properties to the nodes of the graphs. You can define as many symbols as you like. For the Noisy XOR problem, you only need two.

- *Vector Symbolic Representation (Hypervectors).* You also decide how large hypervectors you would like to use to store the symbols. Larger hypervectors room more symbols. Since you only have two symbols, set the size to _32_. Finally, you decide how many bits to use for representing each symbol. Use _2_ bits for this tutorial. You then get _32*31/2 = 496_ unique bit pairs - plenty of space for two symbols!
  
- *Generation and Compilation.* The generation and compilation of hypervectors happen automatically during initialization, using [sparse distributed codes](https://ieeexplore.ieee.org/document/917565).

### Adding Nodes

The next step is to set how many nodes you want in each of the _10,000_ graphs you are building. For the Noisy XOR problem, each graph has two nodes:
```bash
for graph_id in range(10000):
    graphs_train.set_number_of_graph_nodes(graph_id, 2)
```
After doing that, you prepare for adding the nodes:
```bash
graphs_train.prepare_node_configuration()
```
You add the two nodes to the graphs as follows, giving them one outgoing edge each:
```bash
for graph_id in range(10000):
   number_of_outgoing_edges = 1
   graphs_train.add_graph_node(graph_id, 'Node 1', number_of_outgoing_edges)
   graphs_train.add_graph_node(graph_id, 'Node 2', number_of_outgoing_edges)
```

### Adding Edges

You are now ready to prepare for adding edges:
```bash
graphs_train.prepare_edge_configuration()
```

Next, you connect the two nodes of each graph:
```bash
for graph_id in range(10000):
    edge_type = "Plain"
    graphs_train.add_graph_node_edge(graph_id, 'Node 1', 'Node 2', edge_type)
    graphs_train.add_graph_node_edge(graph_id, 'Node 2', 'Node 1', edge_type)
```
You need two edges because you build directed graphs, and with two edges you cover both directions. Use only one edge type, named _Plain_.

### Adding Properties and Class Labels

In the last step, you randomly assign property **A** or **B** to each node.
```bash
Y_train = np.empty(10000, dtype=np.uint32)
for graph_id in range(10000):
    x1 = random.choice(['A', 'B'])
    x2 = random.choice(['A', 'B'])

    graphs_train.add_graph_node_property(graph_id, 'Node 1', x1)
    graphs_train.add_graph_node_property(graph_id, 'Node 2', x2)
```
Based on this assignment, you set the class label of the graph. If both nodes get the same property, the class label is _0_. Otherwise, it is _1_.
```bash
    if x1 == x2:
        Y_train[graph_id] = 0
    else:
        Y_train[graph_id] = 1
```
The class label is finally randomly inverted to introduce noise.
```bash
    if np.random.rand() <= 0.01:
        Y_train[graph_id] = 1 - Y_train[graph_id]
```
