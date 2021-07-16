# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 14:56:25 2020

@author: Eduin Hernandez
"""

import shelve
import numpy as np
import matplotlib.pyplot as plt

#-----------------------------------------------------------------------------
folder_path = 'D:/Dewen/MNIST_FCN/Accuracy/'
suffix = '.out'

prefix = 'shelve_accuracy_mnist_dense_'
tail = ''

tmax = '1'
epochs_num = 3

'UEP Class Num'
class_num = 3

'Coding Type'
operator_str = ['centralized',
                'decentralized',
                'now',
                'ew',
                'block_reps']

'Plotting Variables'

step = 100
legend = ['Paramter Server',
          'Uncoded',
          'NOW - UEP - 3 Classes',
          'EW - UEP - 3 Classes',
          'Block Reps']

ext_ind = [500, 1000, 1500, 2000] #specific indexes to extract


#-----------------------------------------------------------------------------
wait_str = '_tmax' + tmax

title = 'Mnist Classification Accuracy\n Epochs = ' + str(epochs_num) + ', Tmax = ' + tmax

base_str = 'centralized'
uep_str = 'now_ew'


acc = {}
for op, ind in zip(operator_str, range(len(operator_str))):
    if(op == base_str):
        filename_load = prefix + op + str(epochs_num)
    elif(op in uep_str):
        filename_load = prefix + op + str(epochs_num) + '_class' + str(class_num) + wait_str + tail
    else:
        filename_load = prefix + op + str(epochs_num) + wait_str + tail
    
    my_shelf = shelve.open(folder_path + filename_load + suffix)
    acc[ind] = my_shelf['data']['acc'].mean(axis=0)
    my_shelf.close()
 
plt.close('all')
x = np.arange(50, 937*epochs_num, step)
x = np.concatenate((np.array([0]),x, np.array([937*epochs_num-1])))
acc_tilde = np.zeros((len(operator_str), x.size))
acc_ext = np.zeros((len(operator_str), len(ext_ind)))
for i0 in range(len(operator_str)):
    plt.plot(x, acc[i0][x])
    acc_tilde[i0] = acc[i0][x]
    acc_ext[i0] = acc[i0][ext_ind]

acc_tilde = acc_tilde.T
acc_ext = acc_ext.T

# plt.figure()
# x = np.arange(0, 937*epochs_num)
# for i0 in range(len(operator_str)):
#     a = acc[i0][x].reshape(3,-1).mean(axis=1)
#     plt.plot(np.arange(3), a)


plt.title(title)
plt.grid()
# plt.xlabel('Epoch')
plt.xlabel('minibatch')
plt.ylabel('accuracy')
plt.legend(legend, loc = 'lower right')

# np.savetxt('D:/Dewen/text.txt', (x,y), fmt='%.5')
