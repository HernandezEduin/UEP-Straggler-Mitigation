# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 21:15:06 2020

@author: Eduin Hernandez
Summary: MNIST Accuracy Data
"""

import keras

from tqdm import trange
from timeit import default_timer as timer
import shelve
import numpy as np
import matplotlib.pyplot as plt
from Neural_Networks.neural_networks import *

#-----------------------------------------------------------------------------
"Changeable Variables"
batch_size = 60000 #64
num_classes = 10
epoch_num = 100
learning_rate = 0.1
sample_data = 1 #3

mf = 0 #index of model to use
max_wait = 2
lam = 0.5

save_file=False #Save the data set
# filename_save = 'shelve_accuracy_mnist_dense_decentralized3_tmax0.25_col' #Save location and name
folder_path = 'D:/Dewen/Accuracy/'
suffix = '.out'

plot_state = True
#-----------------------------------------------------------------------------

"Training"
def train(network,X,y):
    # Train our network on a given batch of X and y.
    # We first need to run forward to get all layer activations.
    # Then we can run layer.backward going from last to first layer.
    # After we have called backward for all layers, all Dense layers have already made one gradient step.
    
    
    # Get the layer activations
    layer_activations = forward(network,X)
    layer_inputs = [X]+layer_activations  #layer_input[i] is an input for network[i]
    logits = layer_activations[-1]
    
    pred = softmax(logits).argmax(axis=-1)
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
        
    # return np.mean(loss), np.mean(pred==y.argmax(axis=1)), pack
    return np.mean(loss), np.mean(pred==y), pack

"Prediction Model"
def predict(network,X):
    # Compute network predictions. Returning indices of largest Logit probability
    logits = forward(network,X)[-1]
    return logits.argmax(axis=-1)

def predict_batch(network,x,batch_size):
    pred = []
    for x_batch in iterate_prediction(x,batchsize=batch_size):
        pred.append(predict(network, x_batch))
    return np.asarray(pred).flatten()

# def network_performance(network, X, Y):
#     logits = forward(network,X)[-1]
#     acc = logits.argmax(axis=-1) == Y
#     lss = loss_func(logits, Y)
#     return lss, acc

# def network_performance_batch(network, X, Y):
#     accuracy = []
#     loss = []
#     for x_batch, y_batch in iterate_minibatches_verbose_off(X,Y, batch_size):
#         l, a = network_performance(network, x_batch, y_batch)
#         accuracy.append(l)
#         loss.append(a)
#     return np.asarray(loss).flatten(), np.asarray(accuracy).flatten()

def model0(input, num_classes, learning_rate):
    'Uncoded - Centralized'
    network = []
    output_shape = input.shape[1:]
    
    network.append(Flatten())
    output_shape = network[-1].get_output_shape(output_shape)
    
    network.append(Dense(output_shape,100, learning_rate= learning_rate))
    output_shape = network[-1].get_output_shape(output_shape)
    
    network.append(ReLU())
    output_shape = network[-1].get_output_shape(output_shape)
    
    network.append(Dense(output_shape,200, learning_rate= learning_rate))
    output_shape = network[-1].get_output_shape(output_shape)
    
    network.append(ReLU())
    output_shape = network[-1].get_output_shape(output_shape)
    
    network.append(Dense(output_shape,num_classes, learning_rate= learning_rate))
    return network

def model1(input, num_classes, learning_rate):
    'Uncoded - Decentralized - Rows x Cols'
    op_pack = {}
    op_pack['max_workers'] = 9
    op_pack['max_wait'] = max_wait
    op_pack['A_partitions'] = 3
    op_pack['B_partitions'] = 3
    op_pack['lam'] = lam
    op_pack['K'] = 9/op_pack['max_workers']
    
    network = []
    output_shape = input.shape[1:]
    
    network.append(Flatten())
    output_shape = network[-1].get_output_shape(output_shape)
    
    network.append(Dense(output_shape,100, learning_rate= learning_rate, operations = [0,1,1], operation_pack = op_pack))
    output_shape = network[-1].get_output_shape(output_shape)
    
    network.append(ReLU())
    output_shape = network[-1].get_output_shape(output_shape)
    
    network.append(Dense(output_shape,200, learning_rate= learning_rate, operations = [0,1,1], operation_pack = op_pack))
    output_shape = network[-1].get_output_shape(output_shape)
    
    network.append(ReLU())
    output_shape = network[-1].get_output_shape(output_shape)
    
    network.append(Dense(output_shape,num_classes, learning_rate= learning_rate, operations = [0,1,1], operation_pack = op_pack))
    return network

def model2(input, num_classes, learning_rate):
    'UEP - NOW Model - Rows x Cols'
    with open('D:/Dewen/Accuracy/prob_NOW_6classes.csv', 'r', encoding='utf-8-sig') as f: 
        prob = np.genfromtxt(f, dtype=float, delimiter=',')
    
    op_pack = {}
    op_pack['classes_num'] = prob.shape[0]
    op_pack['class_prob'] = prob
    op_pack['max_workers'] = 15
    op_pack['max_wait'] = max_wait
    op_pack['A_partitions'] = 3
    op_pack['B_partitions'] = 3
    op_pack['lam'] = lam
    op_pack['K'] = 9/op_pack['max_workers']
    op_pack['class_table'] = [[0, 1, 3],
                              [1, 2, 4],
                              [3, 4, 5]]
    
    network = []
    output_shape = input.shape[1:]
    
    network.append(Flatten())
    output_shape = network[-1].get_output_shape(output_shape)
    
    network.append(Dense(output_shape,100, learning_rate= learning_rate, operations = [0,2,2], operation_pack = op_pack))
    output_shape = network[-1].get_output_shape(output_shape)
    
    network.append(ReLU())
    output_shape = network[-1].get_output_shape(output_shape)
    
    network.append(Dense(output_shape,200, learning_rate= learning_rate, operations = [0,2,2], operation_pack = op_pack))
    output_shape = network[-1].get_output_shape(output_shape)
    
    network.append(ReLU())
    output_shape = network[-1].get_output_shape(output_shape)
    
    network.append(Dense(output_shape,num_classes, learning_rate= learning_rate, operations = [0,2,2], operation_pack = op_pack))
    return network

def model3(input, num_classes, learning_rate):
    'UEP - NOW Model - Rows x Cols'
    with open('D:/Dewen/Accuracy/prob_NOW_3classes.csv', 'r', encoding='utf-8-sig') as f: 
        prob = np.genfromtxt(f, dtype=float, delimiter=',')
    
    op_pack = {}
    op_pack['classes_num'] = prob.shape[0]
    op_pack['class_prob'] = prob
    op_pack['max_workers'] = 15
    op_pack['max_wait'] = max_wait
    op_pack['A_partitions'] = 3
    op_pack['B_partitions'] = 3
    op_pack['lam'] = lam
    op_pack['K'] = 9/op_pack['max_workers']
    op_pack['class_table'] = [[0, 1, 2],
                              [1, 2, 2],
                              [2, 2, 2]]
    
    network = []
    output_shape = input.shape[1:]
    
    network.append(Flatten())
    output_shape = network[-1].get_output_shape(output_shape)
    
    network.append(Dense(output_shape,100, learning_rate= learning_rate, operations = [0,2,2], operation_pack = op_pack))
    output_shape = network[-1].get_output_shape(output_shape)
    
    network.append(ReLU())
    output_shape = network[-1].get_output_shape(output_shape)
    
    network.append(Dense(output_shape,200, learning_rate= learning_rate, operations = [0,2,2], operation_pack = op_pack))
    output_shape = network[-1].get_output_shape(output_shape)
    
    network.append(ReLU())
    output_shape = network[-1].get_output_shape(output_shape)
    
    network.append(Dense(output_shape,num_classes, learning_rate= learning_rate, operations = [0,2,2], operation_pack = op_pack))
    return network

def model4(input, num_classes, learning_rate):
    'UEP - EW Model - Rows x Cols'
    with open('D:/Dewen/Accuracy/prob_EW_6classes.csv', 'r', encoding='utf-8-sig') as f: 
        prob = np.genfromtxt(f, dtype=float, delimiter=',')
    
    op_pack = {}
    op_pack['classes_num'] = prob.shape[0]
    op_pack['class_prob'] = prob
    op_pack['max_workers'] = 15
    op_pack['max_wait'] = max_wait
    op_pack['A_partitions'] = 3
    op_pack['B_partitions'] = 3
    op_pack['lam'] = lam
    op_pack['K'] = 9/op_pack['max_workers']
    op_pack['class_table'] = [[0, 1, 3],
                              [1, 2, 4],
                              [3, 4, 5]]
    
    network = []
    output_shape = input.shape[1:]
    
    network.append(Flatten())
    output_shape = network[-1].get_output_shape(output_shape)
    
    network.append(Dense(output_shape,100, learning_rate= learning_rate, operations = [0,2,2], operation_pack = op_pack))
    output_shape = network[-1].get_output_shape(output_shape)
    
    network.append(ReLU())
    output_shape = network[-1].get_output_shape(output_shape)
    
    network.append(Dense(output_shape,200, learning_rate= learning_rate, operations = [0,2,2], operation_pack = op_pack))
    output_shape = network[-1].get_output_shape(output_shape)
    
    network.append(ReLU())
    output_shape = network[-1].get_output_shape(output_shape)
    
    network.append(Dense(output_shape,num_classes, learning_rate= learning_rate, operations = [0,2,2], operation_pack = op_pack))
    return network

def model5(input, num_classes, learning_rate):
    'UEP - EW Model - Rows x Cols'
    with open('D:/Dewen/Accuracy/prob_EW_3classes.csv', 'r', encoding='utf-8-sig') as f: 
        prob = np.genfromtxt(f, dtype=float, delimiter=',')
    
    op_pack = {}
    op_pack['classes_num'] = prob.shape[0]
    op_pack['class_prob'] = prob
    op_pack['max_workers'] = 15
    op_pack['max_wait'] = max_wait
    op_pack['A_partitions'] = 3
    op_pack['B_partitions'] = 3
    op_pack['lam'] = lam
    op_pack['K'] = 9/op_pack['max_workers']
    op_pack['class_table'] = [[0, 1, 2],
                              [1, 2, 2],
                              [2, 2, 2]]
    
    network = []
    output_shape = input.shape[1:]
    
    network.append(Flatten())
    output_shape = network[-1].get_output_shape(output_shape)
    
    network.append(Dense(output_shape,100, learning_rate= learning_rate, operations = [0,2,2], operation_pack = op_pack))
    output_shape = network[-1].get_output_shape(output_shape)
    
    network.append(ReLU())
    output_shape = network[-1].get_output_shape(output_shape)
    
    network.append(Dense(output_shape,200, learning_rate= learning_rate, operations = [0,2,2], operation_pack = op_pack))
    output_shape = network[-1].get_output_shape(output_shape)
    
    network.append(ReLU())
    output_shape = network[-1].get_output_shape(output_shape)
    
    network.append(Dense(output_shape,num_classes, learning_rate= learning_rate, operations = [0,2,2], operation_pack = op_pack))
    return network

def model6(input, num_classes, learning_rate):
    'Block Repetition Model - Rows x Cols'
    op_pack = {}
    op_pack['max_workers'] = 18
    op_pack['max_wait'] = max_wait
    op_pack['A_partitions'] = 3
    op_pack['B_partitions'] = 3
    op_pack['lam'] = lam
    op_pack['K'] = 9/op_pack['max_workers']
    op_pack['reps'] = 2
    
    network = []
    output_shape = input.shape[1:]
    
    network.append(Flatten())
    output_shape = network[-1].get_output_shape(output_shape)
    
    network.append(Dense(output_shape,100, learning_rate= learning_rate, operations = [0,3,3], operation_pack = op_pack))
    output_shape = network[-1].get_output_shape(output_shape)
    
    network.append(ReLU())
    output_shape = network[-1].get_output_shape(output_shape)
    
    network.append(Dense(output_shape,200, learning_rate= learning_rate, operations = [0,3,3], operation_pack = op_pack))
    output_shape = network[-1].get_output_shape(output_shape)
    
    network.append(ReLU())
    output_shape = network[-1].get_output_shape(output_shape)
    
    network.append(Dense(output_shape,num_classes, learning_rate= learning_rate, operations = [0,3,3], operation_pack = op_pack))
    return network

def model7(input, num_classes, learning_rate):
    'Uncoded - Decentralized - Cols x Rows'
    op_pack = {}
    op_pack['max_workers'] = 9
    op_pack['max_wait'] = max_wait
    op_pack['partitions'] = 9
    #op_pack['B_partitions'] = 9
    op_pack['lam'] = lam
    op_pack['K'] = 9/op_pack['max_workers']
    
    network = []
    output_shape = input.shape[1:]
    
    network.append(Flatten())
    output_shape = network[-1].get_output_shape(output_shape)
    
    network.append(Dense(output_shape,100, learning_rate= learning_rate, operations = [0,4,4], operation_pack = op_pack))
    output_shape = network[-1].get_output_shape(output_shape)
    
    network.append(ReLU())
    output_shape = network[-1].get_output_shape(output_shape)
    
    network.append(Dense(output_shape,200, learning_rate= learning_rate, operations = [0,4,4], operation_pack = op_pack))
    output_shape = network[-1].get_output_shape(output_shape)
    
    network.append(ReLU())
    output_shape = network[-1].get_output_shape(output_shape)
    
    network.append(Dense(output_shape,num_classes, learning_rate= learning_rate, operations = [0,4,4], operation_pack = op_pack))
    return network

def model8(input, num_classes, learning_rate):
    'Block Repetition Model - Cols x Rows'
    op_pack = {}
    op_pack['max_workers'] = 18
    op_pack['max_wait'] = max_wait
    op_pack['partitions'] = 9
    #op_pack['B_partitions'] = 9
    op_pack['lam'] = lam
    op_pack['K'] = 9/op_pack['max_workers']
    op_pack['reps'] = 2
    
    network = []
    output_shape = input.shape[1:]
    
    network.append(Flatten())
    output_shape = network[-1].get_output_shape(output_shape)
    
    network.append(Dense(output_shape,100, learning_rate= learning_rate, operations = [0,5,5], operation_pack = op_pack))
    output_shape = network[-1].get_output_shape(output_shape)
    
    network.append(ReLU())
    output_shape = network[-1].get_output_shape(output_shape)
    
    network.append(Dense(output_shape,200, learning_rate= learning_rate, operations = [0,5,5], operation_pack = op_pack))
    output_shape = network[-1].get_output_shape(output_shape)
    
    network.append(ReLU())
    output_shape = network[-1].get_output_shape(output_shape)
    
    network.append(Dense(output_shape,num_classes, learning_rate= learning_rate, operations = [0,5,5], operation_pack = op_pack))
    return network

def model9(input, num_classes, learning_rate):
    'UEP - NOW Model - Cols x Rows'
    with open('D:/Dewen/Accuracy/prob_NOW_3classes.csv', 'r', encoding='utf-8-sig') as f: 
        prob = np.genfromtxt(f, dtype=float, delimiter=',')
    
    op_pack = {}
    op_pack['classes_num'] = prob.shape[0]
    op_pack['class_prob'] = prob
    op_pack['max_workers'] = 15
    op_pack['max_wait'] = max_wait
    op_pack['partitions'] = 9
    op_pack['lam'] = lam
    op_pack['K'] = 9/op_pack['max_workers']
    op_pack['class_table'] = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    
    network = []
    output_shape = input.shape[1:]
    
    network.append(Flatten())
    output_shape = network[-1].get_output_shape(output_shape)
    
    network.append(Dense(output_shape,100, learning_rate= learning_rate, operations = [0,6,6], operation_pack = op_pack))
    output_shape = network[-1].get_output_shape(output_shape)
    
    network.append(ReLU())
    output_shape = network[-1].get_output_shape(output_shape)
    
    network.append(Dense(output_shape,200, learning_rate= learning_rate, operations = [0,6,6], operation_pack = op_pack))
    output_shape = network[-1].get_output_shape(output_shape)
    
    network.append(ReLU())
    output_shape = network[-1].get_output_shape(output_shape)
    
    network.append(Dense(output_shape,num_classes, learning_rate= learning_rate, operations = [0,6,6], operation_pack = op_pack))
    return network

def model10(input, num_classes, learning_rate):
    'UEP - EW Model - Cols x Rows'
    with open('D:/Dewen/Accuracy/prob_EW_3classes.csv', 'r', encoding='utf-8-sig') as f: 
        prob = np.genfromtxt(f, dtype=float, delimiter=',')
    
    op_pack = {}
    op_pack['classes_num'] = prob.shape[0]
    op_pack['class_prob'] = prob
    op_pack['max_workers'] = 15
    op_pack['max_wait'] = max_wait
    op_pack['partitions'] = 9
    op_pack['lam'] = lam
    op_pack['K'] = 9/op_pack['max_workers']
    op_pack['class_table'] = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    
    network = []
    output_shape = input.shape[1:]
    
    network.append(Flatten())
    output_shape = network[-1].get_output_shape(output_shape)
    
    network.append(Dense(output_shape,100, learning_rate= learning_rate, operations = [0,6,6], operation_pack = op_pack))
    output_shape = network[-1].get_output_shape(output_shape)
    
    network.append(ReLU())
    output_shape = network[-1].get_output_shape(output_shape)
    
    network.append(Dense(output_shape,200, learning_rate= learning_rate, operations = [0,6,6], operation_pack = op_pack))
    output_shape = network[-1].get_output_shape(output_shape)
    
    network.append(ReLU())
    output_shape = network[-1].get_output_shape(output_shape)
    
    network.append(Dense(output_shape,num_classes, learning_rate= learning_rate, operations = [0,6,6], operation_pack = op_pack))
    return network
    
model_func = {0: model0,
              1: model1,
              2: model2,
              3: model3,
              4: model4,
              5: model5,
              6: model6,
              7: model7,
              8: model8,
              9: model9,
              10: model10}

#-----------------------------------------------------------------------------
if(mf==0):
    operator_str = 'centralized'
elif(mf==1 or mf==7):
    operator_str = 'decentralized'
elif(mf==2 or mf==3 or mf==9):
    operator_str = 'now'
elif(mf==4 or mf==5 or mf==10):
    operator_str = 'ew'
else:
    operator_str = 'block_reps'

if(mf==2 or mf==4):
    class_str = '_class6'
elif(mf==3 or mf==5 or mf==9 or mf==10):
    class_str = '_class3'
else:
    class_str = ''

if(mf>0):
    wait_str = '_tmax' + str(max_wait)
else:
    wait_str = ''

if(mf>6):
    tail = '_col'
else:
    tail = ''

filename_save = 'shelve_accuracy_mnist_dense_'+ operator_str + str(epoch_num) + class_str + wait_str + tail
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

# y_temp = np.zeros((y_train.shape[0], num_classes), dtype='float')
# y_temp[np.arange(y_train.shape[0]),y_train] = 1
# y_train = y_temp.copy()

#------------------------------------------------------------------------------
hist = {}
if batch_size == 60000:
    for md in range(0, sample_data):
        #--------------------------------------------------------------------------
        "Initializing our model"
        network = model_func[mf](x_train, num_classes, learning_rate)
        
        #--------------------------------------------------------------------------
        "Training and extraction of gradients"
        loss = []
        acc = []
        start_time = timer()
        for epoch in trange(0, epoch_num):
            l, a, _ = train(network, x_train, y_train)
            loss.append(l)
            acc.append(a)
        
        hist[md] = {'loss': loss,
                    'acc': acc}
else:
    for md in trange(0, sample_data):
        #----------------------------------------------------------------------
        "Initializing our model"
        network = model_func[mf](x_train, num_classes, learning_rate)
        
        #----------------------------------------------------------------------
        "Training and extraction of gradients"
        loss = []
        acc = []
        
        start_time = timer()
        for epoch in range(epoch_num):
            for x_batch,y_batch in iterate_minibatches_verbose_off(x_train,y_train,batchsize=batch_size,shuffle=False):
                l, a, _ = train(network,x_batch,y_batch)
                
                loss.append(l)
                acc.append(a)
        
        hist[md] = {'loss': loss,
                    'acc': acc}
        
        "Saving Dataset"
        if save_file:   
            my_shelf = shelve.open(folder_path + filename_save + '_hist' + suffix)   
            my_shelf["hist"] = hist
            my_shelf.close()
        
    
elapsed_time = timer() - start_time # in seconds
print("\nTraining Time elapsed: \n", elapsed_time, "s\n", elapsed_time/60,'m\n', elapsed_time/(60*60), 'h\n\n')

dim = len(hist[0]['acc'])
overall_acc = np.zeros((sample_data, dim))
overall_loss = np.zeros((sample_data, dim))
for md in range(sample_data):
    overall_acc[md] = np.array(hist[md]['acc'])
    overall_loss[md] = np.array(hist[md]['loss'])

"Saving Dataset"
if save_file:   
    my_shelf = shelve.open(folder_path + filename_save + suffix)   
    my_shelf['data'] = {'loss': overall_loss,
                        'acc': overall_acc}
    my_shelf.close()
    
if plot_state:
    plt.close('all')
    plt.figure()
    plt.plot(overall_loss.mean(axis=0))
    plt.title('Loss')
    
    plt.figure()
    plt.plot(overall_acc.mean(axis=0))
    plt.title('Acc')