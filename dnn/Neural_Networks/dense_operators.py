# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 19:35:07 2021

@author: Eduin Hernandez
"""

import numpy as np

def uniform_partition(total_size, class_num):
    indexes = np.linspace(0, total_size, class_num + 1, dtype=int)[:,np.newaxis]
    indexes = np.concatenate((indexes[0:-1],indexes[1:]), axis=1)
    return indexes

def uniform_partition_weights(cummulative_weights, class_num):
    div = np.linspace(0, 1, num = class_num + 1)
    indexes = np.zeros_like(div, dtype=int)[:, np.newaxis]
    for i0, i1 in zip(div[1:], range(1, len(div[1:]))):
        indexes[i1] = (cummulative_weights > i0).argmax()
    indexes[-1] = len(cummulative_weights) - 1
    indexes = np.concatenate((indexes[0:-1],indexes[1:]), axis=1)
    return indexes

def geometric_partition(total_size, class_num):
    indexes = (1/2)**np.linspace(class_num+1, 0, class_num+1, dtype=int)
    indexes[0] = 0
    indexes = total_size*indexes[:, np.newaxis]
    indexes = np.concatenate((indexes[0:-1],indexes[1:]), axis=1).astype(int)
    return indexes

def geometric_partition_weights(cummulative_weights, class_num):
    div = 1 - (1/2)**np.linspace(0, class_num+1, class_num+1, dtype=int)
    div[-1] = 1
    indexes = np.zeros_like(div, dtype=int)[:, np.newaxis]
    for i0, i1 in zip(div[1:], range(1, len(div[1:]))):
        indexes[i1] = (cummulative_weights > i0).argmax()
    indexes[-1] = len(cummulative_weights) - 1
    indexes = np.concatenate((indexes[0:-1],indexes[1:]), axis=1)
    return indexes

def depermute(perm):
    size = len(perm)
    deperm = np.ones_like(perm)*(-1)
    for i0 in range(size):
        deperm[i0] = np.where(perm == i0)[0][0]
    return deperm

def mean_other_axes(mat: np.ndarray, ax: int)-> np.ndarray: #mean along all other axes
    other_axes = np.arange(len(mat.shape))
    other_axes = tuple(np.delete(other_axes, ax, axis=None))
    return np.abs(mat).mean(axis=other_axes)

def perm_mean_together(mean1, mean2) -> np.ndarray:
    mean = mean1*mean2
    return np.argsort(mean)[::-1]

def perm_mean(mean) -> np.ndarray:
    return np.argsort(mean)[::-1]

def normalize(vector) -> np.ndarray:
    norm=np.linalg.norm(vector, ord=1)
    if norm==0:
        norm=np.finfo(vector.dtype).eps
    return vector/norm

def cdf(pdf):
    cdf = np.zeros_like(pdf)
    cdf[0] = pdf[0]
    for i0 in range(1, len(pdf)):
        cdf[i0] = cdf[i0-1] + pdf[i0]
    return cdf

def partition_pdf(pdf, bounds):
    class_num = bounds.shape[0]
    part_pdf = np.zeros(class_num)
    for i0 in range(class_num):
        part_pdf[i0] = pdf[slice(bounds[i0, 0], bounds[i0, 1])].sum()
    return part_pdf


def dense_operator0(A, B, op_pack = {}):
    "Normal Matrix Multiplication"
    return A @ B

def dense_operator1(A, B, op_pack = {}):
    "Matrix Multiplication with no Block Repetitions - Row times Columns"
    indexes_A = uniform_partition(A.shape[0], op_pack['A_partitions'])
    indexes_B = uniform_partition(B.shape[1], op_pack['B_partitions'])
    
    arrival_prob = 1 - np.exp(-1*op_pack['lam']*op_pack['K']*op_pack['max_wait'])
    packets_not_received= (arrival_prob < np.random.random(size = op_pack['max_workers'])).reshape(op_pack['A_partitions'], op_pack['B_partitions'])

    ei = np.where(packets_not_received)
    
    C = A @ B
    
    if(len(ei)>0):
        for i1, i2 in zip(ei[0], ei[1]):
            C[slice(indexes_A[i1,0], indexes_A[i1,1]), slice(indexes_B[i2,0], indexes_B[i2,1])] = 0
    
    return C

def dense_operator2(A, B, op_pack = {}):
    "Matrix Multiplication with UEP Coding - Row times Columns"
    mean_A = mean_other_axes(A, 0)
    mean_B = mean_other_axes(B, 1)
    
    perm_a = perm_mean(mean_A)
    perm_b = perm_mean(mean_B)
    
    deperm_a = depermute(perm_a)
    deperm_b = depermute(perm_b)
    
    A_perm = A[perm_a]
    B_perm = B[:, perm_b]
    
    indexes_A = uniform_partition(A_perm.shape[0], op_pack['A_partitions'])
    indexes_B = uniform_partition(B_perm.shape[1], op_pack['B_partitions'])

    arrival_prob = 1 - np.exp(-1*op_pack['lam']*op_pack['K']*op_pack['max_wait'])
    packets_received= (arrival_prob >= np.random.random(size = op_pack['max_workers'])).sum()
    
    prob = op_pack['class_prob'][:, packets_received]
    class_erasure = prob < np.random.random(op_pack['classes_num'])
    class_erasure = np.where(class_erasure)[0]
    
    C_perm = A_perm @ B_perm
    
    for i0 in class_erasure:
        ei = np.where(op_pack['class_table'] == i0)
        if(len(ei)>0):
            for i1, i2 in zip(ei[0],ei[1]):
                C_perm[slice(indexes_A[i1,0], indexes_A[i1,1]), slice(indexes_B[i2,0], indexes_B[i2,1])] = 0
    
    C_perm = C_perm[deperm_a]
    C_perm = C_perm[:, deperm_b]
    return C_perm

def dense_operator3(A, B, op_pack = {}):
    "Matrix Multiplication with Block Repetitions - Row times Columns"
    indexes_A = uniform_partition(A.shape[0], op_pack['A_partitions'])
    indexes_B = uniform_partition(B.shape[1], op_pack['B_partitions'])

    arrival_prob = 1 - np.exp(-1*op_pack['lam']*op_pack['K']*op_pack['max_wait'])
    packets_not_received= (arrival_prob < np.random.random(size = op_pack['max_workers'])).reshape(op_pack['A_partitions'], op_pack['B_partitions'], op_pack['reps'])
    
    
    packets_not_received.prod(axis=2)
    
    ei = np.where(packets_not_received)
    
    C = A @ B
    
    if(len(ei)>0):
        for i1, i2 in zip(ei[0], ei[1]):
            C[slice(indexes_A[i1,0], indexes_A[i1,1]), slice(indexes_B[i2,0], indexes_B[i2,1])] = 0
    
    return C


def dense_operator4(A, B, op_pack = {}):
    "Matrix Multiplication with no Block Repetitions - Columns times Rows"
    indexes = uniform_partition(A.shape[1], op_pack['partitions'])
    
    arrival_prob = 1 - np.exp(-1*op_pack['lam']*op_pack['K']*op_pack['max_wait'])
    packets_not_received = (arrival_prob < np.random.random(size = op_pack['max_workers']))

    ei = np.where(packets_not_received)
    
    C = np.einsum('ij, jk -> jik', A, B)
    
    if(len(ei)>0):
        for i1 in ei[0]:
            C[slice(indexes[i1,0], indexes[i1,1])] = 0    
    return C.sum(axis=0)

def dense_operator5(A, B, op_pack = {}):
    "Matrix Multiplication with Block Repetitions - Columns times Rows"
    indexes = uniform_partition(A.shape[1], op_pack['partitions'])

    arrival_prob = 1 - np.exp(-1*op_pack['lam']*op_pack['K']*op_pack['max_wait'])
    packets_not_received= (arrival_prob < np.random.random(size = op_pack['max_workers'])).reshape(op_pack['partitions'], op_pack['reps'])
    
    
    packets_not_received.prod(axis=1)
    
    ei = np.where(packets_not_received)
    
    C = np.einsum('ij, jk -> jik', A, B)
    
    if(len(ei)>0):
        for i1 in ei[0]:
            C[slice(indexes[i1,0], indexes[i1,1])] = 0    
    return C.sum(axis=0)

def dense_operator6(A, B, op_pack = {}):
    "Matrix Multiplication with UEP Coding - Columns times Rows"
    mean_A = mean_other_axes(A, 1)
    mean_B = mean_other_axes(B, 0)
    
    perm = perm_mean_together(mean_A, mean_B)
    
    A_perm = A[:, perm]
    B_perm = B[perm]
    
    indexes = uniform_partition(A_perm.shape[1], op_pack['partitions'])

    arrival_prob = 1 - np.exp(-1*op_pack['lam']*op_pack['K']*op_pack['max_wait'])
    packets_received= (arrival_prob >= np.random.random(size = op_pack['max_workers'])).sum()
    
    prob = op_pack['class_prob'][:, packets_received]
    class_erasure = prob < np.random.random(op_pack['classes_num'])
    class_erasure = np.where(class_erasure)[0]
    
    C_perm = np.einsum('ij, jk -> jik', A_perm, B_perm)
    
    
    for i0 in class_erasure:
        ei = np.where(op_pack['class_table'] == i0)
        if(len(ei)>0):
            for i1 in ei[0]:
                C_perm[slice(indexes[i1,0], indexes[i1,1])] = 0    
    return C_perm.sum(axis=0)

dense_op = [dense_operator0, dense_operator1, dense_operator2, dense_operator3, dense_operator4, dense_operator5, dense_operator6]