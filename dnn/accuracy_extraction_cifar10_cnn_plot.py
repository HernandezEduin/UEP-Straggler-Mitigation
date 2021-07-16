# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 16:17:35 2021

@author: Eduin Hernandez
"""

import shelve
import numpy as np
import matplotlib.pyplot as plt

#-----------------------------------------------------------------------------
folder_path = 'D:/Dewen/Cifar10_CNN/Accuracy/'
suffix = '.out'

prefix = 'shelve_accuracy_cifar10_cnn_'
tail = ''
tail2 = ['_0', '_1', '_2', '_3']

tmax = '1.0'
epochs_num = 120
epoch_start = 30
epoch_end = 120
epoch_step = 2

'UEP Class Num'
class_num = 3

'Coding Type'
operator_str = [ 'centralized',
                'decentralized',
                'now',
                'ew',
                'block_reps']

'Plotting Variables'

legend = ['No Stragglers',
          'Uncoded',
          'NOW - UEP - 3 Classes',
          'EW - UEP - 3 Classes',
          'Block Reps']


#-----------------------------------------------------------------------------
wait_str = '_tmax' + tmax

if tail == '_col':
    case_str = 'Col x Row'
else:
    case_str = 'Row x Col'

title = 'Cifar10 Classification Accuracy\n Epochs = ' + str(epoch_end) + ', Tmax = ' + tmax + '\n' + case_str

base_str = 'centralized'
uep_str = 'now_ew'


# acc = {}
acc = np.zeros((len(operator_str), len(tail2), epochs_num))
for op, ind0 in zip(operator_str, range(len(operator_str))):
    for t, ind1 in zip(tail2, range(len(tail2))):
        if(op == base_str):
            filename_load = prefix + op + str(epochs_num) + t
        elif(op in uep_str):
            filename_load = prefix + op + str(epochs_num) + '_class' + str(class_num) + wait_str + tail + t
        else:
            filename_load = prefix + op + str(epochs_num) + wait_str + tail + t
        
        print(filename_load)
        my_shelf = shelve.open(folder_path + filename_load + suffix)
        # acc[ind] = my_shelf['data']['acc']
        acc[ind0][ind1] = my_shelf['data']['acc']
        my_shelf.close()
 
#plt.close('all')
plt.figure()
acc2 = []
x = np.arange(epoch_start, epoch_end, epoch_step)
x = np.concatenate((x, [epoch_end-1]))
for i0 in range(len(operator_str)):
    a = acc[i0].mean(axis=0)
    plt.plot(x + 1, a[x])
    acc2.append(a[x])

x += 1
plt.title(title)
plt.grid()
plt.xlabel('Epoch')
plt.ylabel('accuracy')
plt.legend(legend, loc = 'upper left')

#plt.axvline(30, 0, 1, color='brown')

acc2 = np.array(acc2).T
