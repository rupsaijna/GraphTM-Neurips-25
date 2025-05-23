from GraphTsetlinMachine.graphs import Graphs
import numpy as np
from scipy.sparse import csr_matrix
from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine
from time import time
import argparse

import pandas as pd

def print_output_to_file(opfilename, outputstr):
    f = open(opfilename, 'a+')
    f.write('\n-------------')
    f.write('\nParamters:')
    f.write('Epochs:%d, T:%d, s:%.2f, NumClauses:%d,\nHypSize:%d, HypMits:%d,\nDepth:%d, MsgSize:%d, MsgBits:%d,\nMaxSeqLen:%d,MaxIncLit:%d, StateBits:%d'%(epochs,T,s, number_of_clauses,hypervector_size, hypervector_bits,depth,message_size,message_bits,max_sequence_length, max_included_literals, number_of_state_bits))
    f.write('\nResults:')
    f.write(outputstr)
    f.close()


def calc_graph_info(df):
    df['final_state'] = '1:9 2:9 3:9 4:9 5:9'
    calc_state_headers = state_headers[:-1]+['final_state']
    for ind, row in df.iterrows():
        allstates = row[calc_state_headers].tolist()
        uniquestates = []
        node_edge_cnts =[]
        for st in allstates:
            if st not in uniquestates:
                uniquestates.append(st)
        edgedata = row[action_headers].tolist()
        outedgecnts= {nd:0 for nd in range(len(uniquestates))}
        inedgecnts= {nd:0 for nd in range(len(uniquestates))}
        df.loc[ind, 'uniquestatescnt'] = int(len(uniquestates))
        df.loc[ind, 'uniquestates'] = str(uniquestates)
        edges = []
        for ed in range(len(edgedata)):
            fromstate = allstates[ed]
            tostate = allstates[ed+1]
            from_node_id = uniquestates.index(fromstate)
            to_node_id = uniquestates.index(tostate)
            outedgecnts[from_node_id]+= 1
            inedgecnts[to_node_id]+= 1
            edges.append((from_node_id, to_node_id))
        df.loc[ind, 'calculatededges'] = str(edges)
        df.loc[ind, 'outedges'] = str(outedgecnts)
        df.loc[ind, 'inedges'] = str(inedgecnts)
        
    df['uniquestatescnt']=df['uniquestatescnt'].astype(int)

filename = 'data/tangrams-train.tsv'
devfilename = 'data/tangrams-dev.tsv'
opfilename = 'tangrams_3u.txt'

headers = ['rowid', 'state1', 'action1','state2', 'action2', 'state3', 'action3', 'state4', 'action4', 'state5', 'action5', 'state6']


## TM Paramters
epochs=100
number_of_clauses=800
T=9000
s=0.05
number_of_state_bits=8
depth=6
hypervector_size=256
hypervector_bits=4
message_size=256
message_bits=2
double_hashing=False
noise=0.01

max_sequence_length=1000
max_included_literals=40
## TM Paramters


# Read the CSV file
raw_data_inp= pd.read_csv(filename, sep='\t', names=headers)
raw_data_dev= pd.read_csv(devfilename, sep='\t', names=headers)
raw_data_inp =raw_data_inp.dropna(ignore_index=True)
raw_data_dev=raw_data_dev.dropna(ignore_index=True)

'''Remove following for 5-utterance'''
headers_3utt =  ['rowid', 'state1', 'action1','state2', 'action2', 'state3', 'action3', 'state4']
dropheaders = [h for h in headers if h not in headers_3utt]
raw_data_inp = raw_data_inp.drop(columns=dropheaders)
raw_data_dev = raw_data_dev.drop(columns=dropheaders)
headers = headers_3utt
'''Remove for 5-utterance'''
# Read the CSV file

#Process data from files for graph input
max_number_of_nodes = 6 ##Max number of states per example
action_headers = [h for h in headers if 'action' in h]
state_headers = [h for h in headers if 'state' in h]

number_of_classes=6 ##5 possible figures and Nothing
number_of_examples_train=len(raw_data_inp)
number_of_examples_dev=len(raw_data_dev)

qpos_possibilities=['1','2','3','4','5'] ##Query: which image is in postion x? e.g. x=1

calc_graph_info(raw_data_inp)
calc_graph_info(raw_data_dev)

#Process data from files for graph input




print("Creating training data")

##Make Symbols :: combination of position(1-5) and imagenumbers(0-4 and _)
symbols = []
for pos in range(1,6):
    for img in range(0,5):
        symbols.append('%d:%d'%(pos,img))
    symbols.append('%d:_'%pos)
    symbols.append('%d:9'%pos)

## Create train graphs:: holder

graphs_train = Graphs(
    number_of_examples_train,
    symbols=symbols,
    hypervector_size=hypervector_size,
    hypervector_bits=hypervector_bits,
    double_hashing = double_hashing
)

# Create train graphs:: nodes
for graph_id in range(number_of_examples_train):
    graphs_train.set_number_of_graph_nodes(graph_id, raw_data_inp.loc[graph_id, 'uniquestatescnt'])

graphs_train.prepare_node_configuration()

for graph_id in range(number_of_examples_train):
    outedgecnts = eval(raw_data_inp.loc[graph_id, 'outedges'])
    nodenames = eval(raw_data_inp.loc[graph_id, 'uniquestates'])
    for node_id in range(graphs_train.number_of_graph_nodes[graph_id]):
        number_of_edges = outedgecnts[node_id]
        graphs_train.add_graph_node(graph_id, node_id, number_of_edges)
        graphs_train.add_graph_node_properties(graph_id, node_id, nodenames[node_id].split(' '))



# Create train graphs:: edges
graphs_train.prepare_edge_configuration()


for graph_id in range(number_of_examples_train):
    alledges = eval(raw_data_inp.loc[graph_id, 'calculatededges'])
    edgedata = raw_data_inp.loc[graph_id, action_headers].tolist()
    edgenum = 0
    for edge in alledges:
        from_node_id = int(edge[0])
        to_node_id = int(edge[1])
        graphs_train.add_graph_node_edge(graph_id, from_node_id, to_node_id, edgedata[edgenum])
        edgenum+= 1
    

# Create train graphs:: output
##Which figure is in 1st position? 
#Alt :: 2nd/3rd/4th/5th position
qpos = '1'
Y_train = np.empty(number_of_examples_train, dtype=np.uint32)
for graph_id in range(number_of_examples_train):
    ans = raw_data_inp.loc[graph_id, state_headers[-1]].split(' ')
    for answer in ans:
        if qpos+':' in answer: ##Question pertains to x-th position
            ansasint = answer.replace(qpos+':','')
            if ansasint!='_':
                ansasint = int(ansasint)
            else:
                ansasint= 6
            Y_train[graph_id] =  ansasint
            break
        else:
            Y_train[graph_id] = 6
            
graphs_train.encode()
print("Training data produced")


print("Creating Dev data")

# Create dev graphs:: holder

graphs_dev = Graphs(len(raw_data_dev), init_with=graphs_train)
# Create test graphs:: nodes
for graph_id in range(number_of_examples_dev):
    graphs_dev.set_number_of_graph_nodes(graph_id, raw_data_dev.loc[graph_id, 'uniquestatescnt'])

graphs_dev.prepare_node_configuration()

for graph_id in range(number_of_examples_dev):
    outedgecnts = eval(raw_data_dev.loc[graph_id, 'outedges'])
    nodenames = eval(raw_data_dev.loc[graph_id, 'uniquestates'])
    for node_id in range(graphs_dev.number_of_graph_nodes[graph_id]):
        number_of_edges = outedgecnts[node_id]
        graphs_dev.add_graph_node(graph_id, node_id, number_of_edges)
        graphs_dev.add_graph_node_properties(graph_id, node_id, nodenames[node_id].split(' '))


# Create train graphs:: edges
graphs_dev.prepare_edge_configuration()


for graph_id in range(number_of_examples_dev):
    alledges = eval(raw_data_dev.loc[graph_id, 'calculatededges'])
    edgedata = raw_data_dev.loc[graph_id, action_headers].tolist()
    edgenum = 0
    for edge in alledges:
        from_node_id = int(edge[0])
        to_node_id = int(edge[1])
        graphs_dev.add_graph_node_edge(graph_id, from_node_id, to_node_id, edgedata[edgenum])
        edgenum+= 1
    

# Create train graphs:: output
Y_dev = np.empty(number_of_examples_dev, dtype=np.uint32)
for graph_id in range(number_of_examples_dev):
    ans = raw_data_dev.loc[graph_id, state_headers[-1]].split(' ')
    for answer in ans:
        if qpos+':' in answer: 
            ansasint = answer.replace(qpos+':','')
            if ansasint!='_':
                ansasint = int(ansasint)
            else:
                ansasint= 6
            Y_dev[graph_id] =  ansasint
            break
        else:
            Y_dev[graph_id] = 6

graphs_dev.encode()
print("-Dev data complete")

##TM Setup

tm = MultiClassGraphTsetlinMachine(
    number_of_clauses,
    T,
    s,
    number_of_state_bits = number_of_state_bits,
    depth = depth,
    message_size = message_size,
    message_bits = message_bits,
    max_included_literals = max_included_literals,
    double_hashing = double_hashing,
	grid=(16*13,1,1),
	block=(128,1,1)
)

## Training and Test
from sklearn.metrics import precision_recall_fscore_support as score
import statistics
outputstr =''
avg=[]
for i in range(epochs):
    start_training = time()
    tm.fit(graphs_train, Y_train, epochs=1, incremental=True)
    stop_training = time()

    start_testing = time()
    predicted = tm.predict(graphs_dev)
    stop_testing = time()
    result_test = 100*(predicted == Y_dev).mean()
    avg.append(result_test)
    

    precision, recall, fscore, support = score(Y_dev, predicted,average='macro', zero_division=0.0)
    result_train = 100*(tm.predict(graphs_train) == Y_train).mean()

    print("%d Training Accuracy:%.2f Testing Accuracy:%.2f Train Time:%.2f Test Time:%.2f" % (i, result_train, result_test, stop_training-start_training, stop_testing-start_testing))
    print("PRF %.2f %.2f %.2f" % ( precision, recall, fscore))
    
    if i%10 == 0 and i>0:
        outputstr+= "\nEpoch#%d Result_train:%.2f Result_dev:%.2f Avg_dev:%.2f StdDev:%2f Median_dev:%.2f " % (i, result_train, result_test, statistics.mean(avg),statistics.stdev(avg), statistics.median(avg))
        print("\nEpoch#%d Result_train:%.2f Result_dev:%.2f Avg_dev:%.2f StdDev:%2f Median_dev:%.2f " % (i, result_train, result_test, statistics.mean(avg),statistics.stdev(avg), statistics.median(avg)))
    if i%50 == 0 and i>0:
        print('Avg:', statistics.mean(avg),'+/-',statistics.stdev(avg))
        print('Avg Last 10:', statistics.mean(avg[-10:]),'+/-',statistics.stdev(avg[-10:]))
        print('Median:', statistics.median(avg))
        print('Median Last 10:', statistics.median(avg[-10:]))
        outputstr+= "\nEpoch#%d Result_train:%.2f Result_dev:%.2f P:%.2f R:%.2f F1:%.2f " % (i, result_train, result_test, precision, recall, fscore)


print("\nEpoch#%d Result_train:%.2f Result_dev:%.2f Avg_dev:%.2f StdDev:%2f Median_dev:%.2f " % (i, result_train, result_test, statistics.mean(avg),statistics.stdev(avg), statistics.median(avg)))
print('Number of Training examples:', number_of_examples_train)
print('Number of Dev examples:', number_of_examples_dev)
print_output_to_file(opfilename, outputstr)

      
