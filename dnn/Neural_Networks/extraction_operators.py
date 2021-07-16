# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 15:45:24 2021

@author: Eduin Hernandez
"""
import numpy as np
def gradient_save(p, *grads):
    for g0 in grads:
        p.append(g0.copy())

def gradient_pass(p, *grads):
    pass

grad_store_op = [gradient_pass, gradient_save]