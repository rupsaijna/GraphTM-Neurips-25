# Experiment

This is the code to grenerate the results from Figure 7. in section 3.6 in the Multivalue  Noisy XOR experiment in the paper titled [The Tsetlin Machine Goes Deep: Logical Learning and Reasoning With Graphs]. 

## Requirements

The expreiment have the following requirements:

```setup
graphtsetlinmachine==0.3.3
matplotlib==3.8.2
numpy==1.26.2
pandas==2.1.3
seaborn==0.13.0
```
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
