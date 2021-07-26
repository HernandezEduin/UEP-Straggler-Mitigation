# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 15:54:34 2021

@author: Eduin Hernandez

Summary: Simulation of Straggling Mitigation with Protections Codes for Distributed Approximate Matrix Multiplication
        in a Deep Learning Scenario.
Dataset: Cifar10 Data
"""
import os

from keras.datasets import cifar10
from Neural_Networks.neural_networks import *

import argparse
import numpy as np

from timeit import default_timer as timer
import shelve

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
    parser.add_argument('--class-num', type=int, default=10, help='Number of classes for classifying')    
    parser.add_argument('--batch-size', type=float, default=64, help='Batch Size for Training and Testing')
    parser.add_argument('--epoch-num', type=int, default=120, help='End Iterations for Training')
    parser.add_argument('--epoch-start', type=int, default=30, help='Start Iterations for Training')
    parser.add_argument('--learning-rate', type=float, default=0.01, help='Learning Rate for model')
    
    parser.add_argument('--model-index', type=int, default=0 , help='Model to use for the learning. Model 0 for Centralized, 1 and 2 for decentralized for rxc and cxr respectively. 3 and 4 for now. 5 and 6 for ew. 7 and 8 for block reps.')
    
    'Approximation Parameters'
    parser.add_argument('--max-wait', type=float, default = 0.5, help='Maximum wait time for results to arrive')
    parser.add_argument('--lam', type=float, default = 0.5, help='Scaling parameter for delay')
    parser.add_argument('--prob-path', type=str, default ='./', help='Path for probabilities of correct decoding' )
    
    parser.add_argument('--suffix', type=str, default= '.out', help='Save file extension')
    
    'Save Details'
    parser.add_argument('--save-acc', type=str2bool, default='True', help='Whether to Save the weights of the training')
    parser.add_argument('--filename-acc', type=str, default= 'shelve_accuracy_cifar10_cnn_', help='Filename for Saving Accuracy')
    parser.add_argument('--folder-path-acc', type=str, default= './Accuracy/', help='Folder Save file path for Accuracy')
    parser.add_argument('--id', type=str, default='', help='ID value for accuracy')
    
    'Loading Weights'
    parser.add_argument('--load-weights', type=str2bool, default='False', help='Whether to load the model weights at epoch start for the learning')
    parser.add_argument('--filename-weights', type=str, default= 'shelve_weights_cifar10_cnn_epoch', help='Filename for Saving Weights')
    parser.add_argument('--folder-path-weights', type=str, default='./Weights/', help='Folder Save file path for Weights')
    
    args = parser.parse_args()
    return args
#-----------------------------------------------------------------------------
"Training"
def train(network,X,y):
    # Train our network on a given batch of X and y.
    
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

#-----------------------------------------------------------------------------
'Base Model'
def deep_model(input, num_classes, learning_rate, operations = np.zeros((3,3), dtype=int), op_pack = [[{},{},{}], [{},{},{}], [{},{},{}]]):
    network = []
    output_shape = input.shape[1:]
    
    network.append(Conv2d(32, (3,3), padding='same', input_shape = output_shape,
                          learning_rate= learning_rate))
    output_shape = network[-1].get_output_shape(output_shape)

    network.append(ReLU())
    output_shape = network[-1].get_output_shape(output_shape)
    
    network.append(Conv2d(32, (3,3), padding='valid', input_shape = output_shape,
                          learning_rate= learning_rate))
    output_shape = network[-1].get_output_shape(output_shape)

    network.append(ReLU())
    output_shape = network[-1].get_output_shape(output_shape)
    
    network.append(MaxPool2d((2,2)))
    output_shape = network[-1].get_output_shape(output_shape)



    "Fully Connected Layers"    
    network.append(Flatten())
    output_shape = network[-1].get_output_shape(output_shape)

   
    network.append(Dense(output_shape,512, learning_rate= learning_rate, operations = operations[0], operation_pack = op_pack[0]))
    output_shape = network[-1].get_output_shape(output_shape)

    
    network.append(ReLU())
    output_shape = network[-1].get_output_shape(output_shape)
    
    network.append(Dense(output_shape,256, learning_rate= learning_rate, operations = operations[1], operation_pack = op_pack[1]))
    output_shape = network[-1].get_output_shape(output_shape)

    
    network.append(ReLU())
    output_shape = network[-1].get_output_shape(output_shape)
    
    network.append(Dense(output_shape,num_classes, learning_rate= learning_rate, operations = operations[2], operation_pack = op_pack[2]))

    
    if(args.load_weights):
        my_shelf = shelve.open(args.folder_path_weights + args.filename_weights +  str(args.epoch_start) + args.suffix)   
        weights = my_shelf['weights']
        my_shelf.close()

        for l, w in zip(network, weights):
            l.set_weights(*w)
    
    return network

#-----------------------------------------------------------------------------
def prepare_op_packs(operations, op_pack, op_unc):
    op_packs = []
    for op0 in operations:
        o_p = []
        for op1 in op0:
            if op1 == 0:
                o_p.append({})
            elif op1 == 1 or op1== 4:
                o_p.append(op_unc)
            else:
                o_p.append(op_pack)
        op_packs.append(o_p.copy())
    return op_packs

def model0(input, args, op_unc):
    'Uncoded - Centralized'
    return deep_model(input, args.class_num, args.learning_rate)

def model1(input, args, op_unc):
    'Uncoded - Decentralized - Rows x Cols'
    
    operations = [[0,1,1],
                  [0,1,1],
                  [0,1,1]]
    
    op_packs = prepare_op_packs(operations, op_unc, op_unc)
    
    return deep_model(input, args.class_num, args.learning_rate, operations, op_packs)

def model2(input, args, op_unc):
    'Uncoded - Decentralized - Cols x Rows'
    
    operations = [[0,4,4],
                  [0,4,4],
                  [0,4,4]]
    
    op_packs = prepare_op_packs(operations, op_unc, op_unc)
    
    return deep_model(input, args.class_num, args.learning_rate, operations, op_packs)

def model3(input, args, op_unc):
    'UEP - NOW Model - Rows x Cols'
    with open(args.prob_path + 'prob_NOW_3classes.csv', 'r', encoding='utf-8-sig') as f: 
        prob = np.genfromtxt(f, dtype=float, delimiter=',')
    
    op_pack = {}
    op_pack['classes_num'] = prob.shape[0]
    op_pack['class_prob'] = prob
    op_pack['max_workers'] = 15
    op_pack['max_wait'] = args.max_wait
    op_pack['A_partitions'] = 3
    op_pack['B_partitions'] = 3
    op_pack['lam'] = args.lam
    op_pack['K'] = op_unc['max_workers']/op_pack['max_workers']
    op_pack['class_table'] = [[0, 1, 2],
                              [1, 2, 2],
                              [2, 2, 2]]
    
    operations = [[0,2,2],
                  [0,2,2],
                  [0,2,1]]
    
    op_packs = prepare_op_packs(operations, op_pack, op_unc)
    
    return deep_model(input, args.class_num, args.learning_rate, operations, op_packs)
        
def model4(input, args, op_unc):
    'UEP - NOW Model - Cols x Rows'
    with open(args.prob_path + 'prob_NOW_3classes.csv', 'r', encoding='utf-8-sig') as f: 
        prob = np.genfromtxt(f, dtype=float, delimiter=',')
    
    op_pack = {}
    op_pack['classes_num'] = prob.shape[0]
    op_pack['class_prob'] = prob
    op_pack['max_workers'] = 15
    op_pack['max_wait'] = args.max_wait
    op_pack['partitions'] = 9
    op_pack['lam'] = args.lam
    op_pack['K'] = op_unc['max_workers']/op_pack['max_workers']
    op_pack['class_table'] = [0, 1, 1, 2, 2, 2, 2, 2, 2]
    
    operations = [[0,6,6],
                  [0,6,6],
                  [0,4,4]]
    
    op_packs = prepare_op_packs(operations, op_pack, op_unc)
    
    return deep_model(input, args.class_num, args.learning_rate, operations, op_packs)
        
def model5(input, args, op_unc):
    'UEP - EW Model - Rows x Cols'
    with open(args.prob_path + 'prob_EW_3classes.csv', 'r', encoding='utf-8-sig') as f: 
        prob = np.genfromtxt(f, dtype=float, delimiter=',')
    
    op_pack = {}
    op_pack['classes_num'] = prob.shape[0]
    op_pack['class_prob'] = prob
    op_pack['max_workers'] = 15
    op_pack['max_wait'] = args.max_wait
    op_pack['A_partitions'] = 3
    op_pack['B_partitions'] = 3
    op_pack['lam'] = args.lam
    op_pack['K'] = op_unc['max_workers']/op_pack['max_workers']
    op_pack['class_table'] = [[0, 1, 2],
                              [1, 2, 2],
                              [2, 2, 2]]
    
    operations = [[0,2,2],
                  [0,2,2],
                  [0,2,1]]
    
    op_packs = prepare_op_packs(operations, op_pack, op_unc)
    
    return deep_model(input, args.class_num, args.learning_rate, operations, op_packs)

def model6(input, args, op_unc):
    'UEP - EW Model - Cols x Rows'
    with open(args.prob_path + 'prob_EW_3classes.csv', 'r', encoding='utf-8-sig') as f: 
        prob = np.genfromtxt(f, dtype=float, delimiter=',')
    
    op_pack = {}
    op_pack['classes_num'] = prob.shape[0]
    op_pack['class_prob'] = prob
    op_pack['max_workers'] = 15
    op_pack['max_wait'] = args.max_wait
    op_pack['partitions'] = 9
    op_pack['lam'] = args.lam
    op_pack['K'] = op_unc['max_workers']/op_pack['max_workers']
    op_pack['class_table'] = [0, 1, 1, 2, 2, 2, 2, 2, 2]
    
    operations = [[0,6,6],
                  [0,6,6],
                  [0,4,4]]
    
    op_packs = prepare_op_packs(operations, op_pack, op_unc)
    
    return deep_model(input, args.class_num, args.learning_rate, operations,  op_packs)

def model7(input, args, op_unc):
    'Block Repetition Model - Rows x Cols'
    op_pack = {}
    op_pack['max_workers'] = 18
    op_pack['max_wait'] = args.max_wait
    op_pack['A_partitions'] = 3
    op_pack['B_partitions'] = 3
    op_pack['lam'] = args.lam
    op_pack['K'] = op_unc['max_workers']/op_pack['max_workers']
    op_pack['reps'] = 2
    
    operations = [[0,3,3],
                  [0,3,3],
                  [0,3,1]]
    
    op_packs = prepare_op_packs(operations, op_pack, op_unc)
    
    return deep_model(input, args.class_num, args.learning_rate, operations, op_packs)

def model8(input, args, op_unc):
    'Block Repetition Model - Cols x Rows'
    op_pack = {}
    op_pack['max_workers'] = 18
    op_pack['max_wait'] = args.max_wait
    op_pack['partitions'] = 9
    op_pack['lam'] = args.lam
    op_pack['K'] = op_unc['max_workers']/op_pack['max_workers']
    op_pack['reps'] = 2
    
    operations = [[0,5,5],
                  [0,5,5],
                  [0,4,4]]
    
    op_packs = prepare_op_packs(operations, op_pack, op_unc)
    
    return deep_model(input, args.class_num, args.learning_rate, operations, op_packs)

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

filename_save = args.filename_acc + operator_str + str(args.epoch_num) + class_str + wait_str + tail

if not os.path.exists(args.folder_path_acc):
    os.makedirs(args.folder_path_acc)
    
if not os.path.exists(args.folder_path_weights):
    os.makedirs(args.folder_path_weights)
#-----------------------------------------------------------------------------
op_pack = {}
op_pack['max_workers'] = 9
op_pack['max_wait'] = args.max_wait
op_pack['A_partitions'] = 3
op_pack['B_partitions'] = 3
op_pack['partitions'] = op_pack['A_partitions']*op_pack['B_partitions']
op_pack['lam'] = args.lam
op_pack['K'] = 9/ op_pack['max_workers']
#-----------------------------------------------------------------------------
"Data Loading"
# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

#-----------------------------------------------------------------------------
"Data Preprocessing"
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

y_train = y_train.reshape(-1)
y_test = y_test.reshape(-1)
#-----------------------------------------------------------------------------
"Initializing our model"
network = model_func[args.model_index](x_train, args, op_pack)

#-----------------------------------------------------------------------------
"Training and extraction of gradients"
loss = []
acc = []

if args.load_weights:
    my_shelf = shelve.open(args.folder_path_acc + args.filename_acc[0:-1] + args.suffix)   
    acc = my_shelf['data']['acc'][0:args.epoch_start]
    loss = my_shelf['data']['loss'][0:args.epoch_start]
    my_shelf.close()

start_time = timer()

for epoch in range(args.epoch_start, args.epoch_num):
    a0 = []
    l0 = []

    for x_batch,y_batch, t in iterate_minibatches(x_train,y_train,batchsize=args.batch_size,shuffle=True):
        l1, a1, _ = train(network,x_batch,y_batch)
        l0.append(l1)
        a0.append(a1)
        
        t.set_description("Epoch: " + str(epoch + 1) + '/' + str(args.epoch_num) + " Acc:" + str(f'{np.mean(a0): 0.3f}') + " Loss:" +  str(f'{np.mean(l0): 0.3f}'))

    loss.append(np.mean(l0))
    acc.append(np.mean(a0))
    
    t.close()
    

elapsed_time = timer() - start_time # in seconds
print("\nTraining Time elapsed: \n", elapsed_time, "s\n", elapsed_time/60,'m\n', elapsed_time/(60*60), 'h\n\n')

overall_acc = np.asarray(acc).flatten()
overall_loss = np.asarray(loss).flatten()

"Saving Dataset"
if args.save_acc:   
    my_shelf = shelve.open(args.folder_path_acc + filename_save + args.id + args.suffix)   
    my_shelf['data'] = {'loss': overall_loss,
                        'acc': overall_acc}
    my_shelf.close()