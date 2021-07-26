# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 21:15:06 2020

@author: Eduin Hernandez

Summary: Simulation of Straggling Mitigation with Protections Codes for Distributed Approximate Matrix Multiplication
        in a Deep Learning Scenario.
Dataset: MNIST Data
"""
import os

import keras
from Neural_Networks.neural_networks import *

from tqdm import trange
from timeit import default_timer as timer
import shelve

import argparse
import numpy as np
import matplotlib.pyplot as plt


def str2bool(string):
    if isinstance(string, bool):
       return string
   
    if string.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif string.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
    parser = argparse.ArgumentParser(description='Variables for Cifar10 Training')

    'Model Details'
    parser.add_argument('--class-num', type=int, default=10, help='Number of classes to classify')    
    parser.add_argument('--batch-size', type=float, default=64, help='Batch Size for Training and Testing')
    parser.add_argument('--epoch-num', type=int, default=3, help='Number of Iterations for Training')
    parser.add_argument('--learning-rate', type=float, default=0.01, help='Learning Rate for model')
    
    parser.add_argument('--model-index', type=int, default = 0, help='Model to use for the learning. Model 0 for Centralized, 1 and 2 for decentralized for rxc and cxr respectively. 3 and 4 for now. 5 and 6 for ew. 7 and 8 for block reps.')
    parser.add_argument('--sample-num', type=int, default = 1, help='Number of Models to train from scratch. The more models, better statistics.')
    
    'Approximation Parameters'
    parser.add_argument('--max-wait', type=float, default = 0.5, help='Maximum wait time for results to arrive')
    parser.add_argument('--lam', type=float, default = 0.5, help='Scaling parameter for delay')
    parser.add_argument('--prob-path', type=str, default ='./', help='Path for probabilities of correct decoding for UEP in csv file.' )
    
    parser.add_argument('--suffix', type=str, default= '.out', help='Save file extension')
    
    'Save Details'
    parser.add_argument('--save-acc', type=str2bool, default='True', help='Whether to Save the weights of the training')
    parser.add_argument('--filename-acc', type=str, default= 'shelve_accuracy_mnist_', help='Filename for Saving Accuracy')
    parser.add_argument('--folder-path-acc', type=str, default= './Accuracy/', help='Folder Save file path for Accuracy')

    
    'Plot'
    parser.add_argument('--plot-state', type=str2bool, default='True', help='Whether to Plot the Accuracy and Loss')

    args = parser.parse_args()
    return args

#-----------------------------------------------------------------------------

"Training"
def train(network,X,y):
    # Train our network on a given batch of X and Y
    
    # Get the layer activations
    layer_activations = forward(network,X)
    layer_inputs = [X]+layer_activations  #layer_input[i] is an input for network[i]
    logits = layer_activations[-1]
    
    pred = logits.argmax(axis=-1)
    # Compute the loss and the initial gradient
    loss = softmax_crossentropy_with_logits(logits,y)
    loss_grad = grad_softmax_crossentropy_with_logits(logits,y)
    

    # Propagate gradients through the network
    # Reverse propogation as this is backprop
    pack = []
    for layer_index in range(len(network))[::-1]:
        layer = network[layer_index]
        
        loss_grad, p = layer.backward(layer_inputs[layer_index],loss_grad) #grad w.r.t. input, also weight updates
        if(len(p)!=0):
            pack.append(p)
        
    return np.mean(loss), np.mean(pred==y), pack

'Base Model'
def deep_model(input, num_classes, learning_rate, operations = np.zeros((3,3), dtype=int), op_pack = [{},{},{}]):
    network = []
    
    output_shape = input.shape[1:]
    
    network.append(Flatten())
    output_shape = network[-1].get_output_shape(output_shape)
    
    network.append(Dense(output_shape,100, learning_rate= learning_rate, operations = operations[0], operation_pack = op_pack))
    output_shape = network[-1].get_output_shape(output_shape)
    
    network.append(ReLU())
    output_shape = network[-1].get_output_shape(output_shape)
    
    network.append(Dense(output_shape,200, learning_rate= learning_rate, operations = operations[1], operation_pack = op_pack))
    output_shape = network[-1].get_output_shape(output_shape)
    
    network.append(ReLU())
    output_shape = network[-1].get_output_shape(output_shape)
    
    network.append(Dense(output_shape,num_classes, learning_rate= learning_rate, operations = operations[2], operation_pack = op_pack))
    return network
    
    
def model0(input, num_classes, learning_rate):
    'Uncoded - Centralized'
    return deep_model(input, num_classes, learning_rate = learning_rate)

def model1(input, num_classes, learning_rate):
    'Uncoded - Decentralized - Rows x Cols'
    op_pack = {}
    op_pack['max_workers'] = 9
    op_pack['max_wait'] = args.max_wait
    op_pack['A_partitions'] = 3
    op_pack['B_partitions'] = 3
    op_pack['lam'] = args.lam
    op_pack['K'] = 9/op_pack['max_workers']
    
    operations = [[0,1,1],
                  [0,1,1],
                  [0,1,1]]
    
    op_package = [{}, op_pack, op_pack]
    
    return deep_model(input, num_classes, learning_rate = learning_rate, operations = operations, op_pack=op_package)

def model2(input, num_classes, learning_rate):
    'Uncoded - Decentralized - Cols x Rows'
    op_pack = {}
    op_pack['max_workers'] = 9
    op_pack['max_wait'] = args.max_wait
    op_pack['partitions'] = 9
    #op_pack['B_partitions'] = 9
    op_pack['lam'] = args.lam
    op_pack['K'] = 9/op_pack['max_workers']
    
    operations = [[0,4,4],
                  [0,4,4],
                  [0,4,4]]
    
    op_package = [{}, op_pack, op_pack]
    
    return deep_model(input, num_classes, learning_rate = learning_rate, operations = operations, op_pack=op_package)

def model3(input, num_classes, learning_rate):
    'UEP - NOW Model - Rows x Cols'
    with open('./prob_NOW_3classes.csv', 'r', encoding='utf-8-sig') as f: 
        prob = np.genfromtxt(f, dtype=float, delimiter=',')
    
    op_pack = {}
    op_pack['classes_num'] = prob.shape[0]
    op_pack['class_prob'] = prob
    op_pack['max_workers'] = 15
    op_pack['max_wait'] = args.max_wait
    op_pack['A_partitions'] = 3
    op_pack['B_partitions'] = 3
    op_pack['lam'] = args.lam
    op_pack['K'] = 9/op_pack['max_workers']
    op_pack['class_table'] = [[0, 1, 2],
                              [1, 2, 2],
                              [2, 2, 2]]
    
    operations = [[0,2,2],
                  [0,2,2],
                  [0,2,2]]
    
    op_package = [{}, op_pack, op_pack]
    
    return deep_model(input, num_classes, learning_rate = learning_rate, operations = operations, op_pack=op_package)

def model4(input, num_classes, learning_rate):
    'UEP - NOW Model - Cols x Rows'
    with open('./prob_NOW_3classes.csv', 'r', encoding='utf-8-sig') as f: 
        prob = np.genfromtxt(f, dtype=float, delimiter=',')
    
    op_pack = {}
    op_pack['classes_num'] = prob.shape[0]
    op_pack['class_prob'] = prob
    op_pack['max_workers'] = 15
    op_pack['max_wait'] = args.max_wait
    op_pack['partitions'] = 9
    op_pack['lam'] = args.lam
    op_pack['K'] = 9/op_pack['max_workers']
    op_pack['class_table'] = [0, 1, 1, 2, 2, 2, 2, 2, 2]
    
    operations = [[0,6,6],
                  [0,6,6],
                  [0,6,6]]
    
    op_package = [{}, op_pack, op_pack]
    

def model5(input, num_classes, learning_rate):
    'UEP - EW Model - Rows x Cols'
    with open('./prob_EW_3classes.csv', 'r', encoding='utf-8-sig') as f: 
        prob = np.genfromtxt(f, dtype=float, delimiter=',')
    
    op_pack = {}
    op_pack['classes_num'] = prob.shape[0]
    op_pack['class_prob'] = prob
    op_pack['max_workers'] = 15
    op_pack['max_wait'] = args.max_wait
    op_pack['A_partitions'] = 3
    op_pack['B_partitions'] = 3
    op_pack['lam'] = args.lam
    op_pack['K'] = 9/op_pack['max_workers']
    op_pack['class_table'] = [[0, 1, 2],
                              [1, 2, 2],
                              [2, 2, 2]]
    
    operations = [[0,2,2],
                  [0,2,2],
                  [0,2,2]]
    
    op_package = [{}, op_pack, op_pack]
    
    return deep_model(input, num_classes, learning_rate = learning_rate, operations = operations, op_pack=op_package)


    return deep_model(input, num_classes, learning_rate = learning_rate, operations = operations, op_pack=op_package)

def model6(input, num_classes, learning_rate):
    'UEP - EW Model - Cols x Rows'
    with open('./prob_EW_3classes.csv', 'r', encoding='utf-8-sig') as f: 
        prob = np.genfromtxt(f, dtype=float, delimiter=',')
    
    op_pack = {}
    op_pack['classes_num'] = prob.shape[0]
    op_pack['class_prob'] = prob
    op_pack['max_workers'] = 15
    op_pack['max_wait'] = args.max_wait
    op_pack['partitions'] = 9
    op_pack['lam'] = args.lam
    op_pack['K'] = 9/op_pack['max_workers']
    op_pack['class_table'] = [0, 1, 1, 2, 2, 2, 2, 2, 2]
    
    operations = [[0,6,6],
                  [0,6,6],
                  [0,6,6]]
    
    op_package = [{}, op_pack, op_pack]
    
    return deep_model(input, num_classes, learning_rate = learning_rate, operations = operations, op_pack=op_package)

def model7(input, num_classes, learning_rate):
    'Block Repetition Model - Rows x Cols'
    op_pack = {}
    op_pack['max_workers'] = 18
    op_pack['max_wait'] = args.max_wait
    op_pack['A_partitions'] = 3
    op_pack['B_partitions'] = 3
    op_pack['lam'] = args.lam
    op_pack['K'] = 9/op_pack['max_workers']
    op_pack['reps'] = 2
    
    operations = [[0,3,3],
                  [0,3,3],
                  [0,3,3]]
    
    op_package = [{}, op_pack, op_pack]
    
    return deep_model(input, num_classes, learning_rate = learning_rate, operations = operations, op_pack=op_package)

def model8(input, num_classes, learning_rate):
    'Block Repetition Model - Cols x Rows'
    op_pack = {}
    op_pack['max_workers'] = 18
    op_pack['max_wait'] = args.max_wait
    op_pack['partitions'] = 9
    #op_pack['B_partitions'] = 9
    op_pack['lam'] = args.lam
    op_pack['K'] = 9/op_pack['max_workers']
    op_pack['reps'] = 2
    
    operations = [[0,5,5],
                  [0,5,5],
                  [0,5,5]]
    
    op_package = [{}, op_pack, op_pack]
    
    return deep_model(input, num_classes, learning_rate = learning_rate, operations = operations, op_pack=op_package)


    
model_func = {0: model0,
              1: model1,
              2: model2,
              3: model3,
              4: model4,
              5: model5,
              6: model6,
              7: model7,
              8: model8}

#-----------------------------------------------------------------------------
args = parse_args()

if(args.model_index==0):
    operator_str = 'centralized'
elif(args.model_index==1 or args.model_index==2):
    operator_str = 'uncoded'
elif(args.model_index==3 or args.model_index==4):
    operator_str = 'now'
elif(args.model_index==5 or args.model_index==6):
    operator_str = 'ew'
else:
    operator_str = 'block_reps'

if(args.model_index>2 and args.model_index<7):
    class_str = '_class3'
else:
    class_str = ''

if(args.model_index>0):
    wait_str = '_tmax' + str(args.max_wait)
else:
    wait_str = ''

if((args.model_index%2)==0 and args.model_index>0):
    tail = '_col'
else:
    tail = ''

filename_save = 'shelve_accuracy_mnist_dense_'+ operator_str + str(args.epoch_num) + class_str + wait_str + tail

if not os.path.exists(args.folder_path_acc):
    os.makedirs(args.folder_path_acc)
#-----------------------------------------------------------------------------
"Data Loading"
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

#-----------------------------------------------------------------------------
"Preprocessing of the Data"
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

y_train = y_train.reshape(-1)
y_test = y_test.reshape(-1)

#------------------------------------------------------------------------------
hist = {}
if args.batch_size == 60000: #Training without Batches
    for md in range(0, args.sample_num):
        #--------------------------------------------------------------------------
        "Initializing our model"
        network = model_func[args.model_index](x_train, args.class_num, args.learning_rate)
        
        #--------------------------------------------------------------------------
        "Training and extraction of gradients"
        loss = []
        acc = []
        start_time = timer()
        for epoch in trange(0, args.epoch_num):
            l, a, _ = train(network, x_train, y_train)
            loss.append(l)
            acc.append(a)
        
        hist[md] = {'loss': loss,
                    'acc': acc}
else: #Training in Batches
    for md in trange(0, args.sample_num):
        #----------------------------------------------------------------------
        "Initializing our model"
        network = model_func[args.model_index](x_train, args.class_num, args.learning_rate)
        
        #----------------------------------------------------------------------
        "Training and extraction of gradients"
        loss = []
        acc = []
        
        start_time = timer()
        for epoch in range(args.epoch_num):
            for x_batch,y_batch in iterate_minibatches_verbose_off(x_train,y_train,batchsize=args.batch_size,shuffle=False):
                l, a, _ = train(network,x_batch,y_batch)
                
                loss.append(l)
                acc.append(a)
        
        hist[md] = {'loss': loss,
                    'acc': acc}
        
    
elapsed_time = timer() - start_time # in seconds
print("\nTraining Time elapsed: \n", elapsed_time, "s\n", elapsed_time/60,'m\n', elapsed_time/(60*60), 'h\n\n')

dim = len(hist[0]['acc'])
overall_acc = np.zeros((args.sample_num, dim))
overall_loss = np.zeros((args.sample_num, dim))
for md in range(args.sample_num):
    overall_acc[md] = np.array(hist[md]['acc'])
    overall_loss[md] = np.array(hist[md]['loss'])

"Saving Dataset"
if args.save_acc:   
    my_shelf = shelve.open(args.folder_path_acc + filename_save + args.suffix)   
    my_shelf['data'] = {'loss': overall_loss,
                        'acc': overall_acc}
    my_shelf.close()
    
if args.plot_state:
    plt.close('all')
    plt.figure()
    plt.plot(overall_loss.mean(axis=0))
    plt.title('Loss')
    
    plt.figure()
    plt.plot(overall_acc.mean(axis=0))
    plt.title('Acc')