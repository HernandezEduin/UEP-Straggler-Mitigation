# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 15:00:57 2020

@author: Eduin Hernandez
"""
import numpy as np ## For numerical python
from skimage.util import view_as_windows as viewW
from tqdm import trange

from Neural_Networks.dense_operators import dense_op
from Neural_Networks.convolution_operators import im2col_4d, conv2d_foward
from Neural_Networks.extraction_operators import grad_store_op


"""Classes for the Layers including fowards and backward propagation"""
"Activation Layers"
class ReLU():
    def __init__(self):
        # ReLU layer simply applies elementwise rectified linear unit to all inputs
        pass
    
    def forward(self, input):
        # Apply elementwise ReLU to [batch, input_units] matrix
        relu_forward = np.maximum(0,input)
        return relu_forward
    
    def backward(self, input, grad_output):
        # Compute gradient of loss w.r.t. ReLU input
        pack = []
        relu_grad = input > 0
        return grad_output*relu_grad, pack
    
    def get_output_shape(self, prev_output_shape):
        return prev_output_shape
    
    def get_weights(self):
        return []
    
    def set_weights(self, *weights):
        pass

class Sigmoid():
    def __init__(self):
        pass
    
    def forward(self, input):
        return 1.0 / (1 + np.exp(-input))

    def backward(self, input, grad_output):
        pack = []
        sig_grad = self.forward(input)*(1-self.forward(input))
        return grad_output*sig_grad, pack
    
    def get_output_shape(self, prev_output_shape):
        return prev_output_shape
    
    def get_weights(self):
        return []
    
    def set_weights(self, *weights):
        pass

"""Extra Layers"""
class Flatten():
    def __init__(self):    
        pass
    
    def forward(self, input):
        flatten_size = np.asarray(input.shape[1:]).prod()
        return input.reshape(-1, flatten_size)

    def backward(self, input, grad_output):
        pack = []
        return grad_output.reshape(input.shape), pack
    
    def get_output_shape(self, prev_output_shape):
        return np.asarray(prev_output_shape).prod()
    
    def get_weights(self):
        return []
    
    def set_weights(self, *weights):
        pass

class DropOut():
    def __init__(self, prob):
        self.p = 1 - prob
    
    def forward(self, input):
        self.drop = np.random.binomial(1,self.p, size=input.shape)
        return self.drop * input

    def backward(self, input, grad_output):
        pack = []
        return self.drop*grad_output, pack
    
    def get_output_shape(self, prev_output_shape):
        return prev_output_shape
    
    def get_weights(self):
        return []
    
    def set_weights(self, *weights):
        pass

"""Main Layers"""
class Dense():
    def __init__(self, input_units, output_units, learning_rate=0.1, operations = [0, 0, 0], operation_pack = [{},{},{}], save_grad=False):      
        self.learning_rate = learning_rate
        self.weights = np.random.normal(loc=0.0, 
                                        scale = np.sqrt(2/(input_units+output_units)), 
                                        size = (input_units,output_units))
        self.biases = np.zeros(output_units)
        
        self.operation_foward = dense_op[operations[0]]
        self.operation_back1 = dense_op[operations[1]]
        self.operation_back2 = dense_op[operations[2]]
        self.op_pack_foward = operation_pack[0]
        self.op_pack_back1 = operation_pack[1]
        self.op_pack_back2 = operation_pack[2]
        self.grad_store_op = grad_store_op[save_grad]
        
    def forward(self, input):
        # return input @ self.weights + self.biases
        return self.operation_foward(input, self.weights, self.op_pack_foward) + self.biases
    
    def backward(self,input,grad_output):
        pack = []
        # compute d f / d x = d f / d dense * d dense / d x
        # where d dense/ d x = weights transposed
        # grad_input = grad_output @ self.weights.T
        grad_input = self.operation_back1(grad_output, self.weights.T, self.op_pack_back1)
        self.grad_store_op(pack, grad_input, grad_output, self.weights.T)
        
        # compute gradient w.r.t. weights and biases
        # grad_weights = input.T @ grad_output
        grad_weights = self.operation_back2(input.T, grad_output, self.op_pack_back2)
        self.grad_store_op(pack, grad_weights, input.T, grad_output)
        
        grad_biases = grad_output.mean(axis=0)*input.shape[0]
        
        assert grad_weights.shape == self.weights.shape and grad_biases.shape == self.biases.shape
        
        # Here we perform a stochastic gradient descent step. 
        self.weights = self.weights - self.learning_rate * grad_weights
        self.biases = self.biases - self.learning_rate * grad_biases
        
        return grad_input, pack
    
    def set_weights(self, *weights):
        self.weights = weights[0].copy()
        self.biases = weights[1].copy()

    def get_weights(self):
        return [self.weights, self.biases]
    
    def get_output_shape(self, prev_output_shape=None):
        return self.weights.shape[1]
    
class MaxPool2d():
    def __init__(self, pooling_shape):
        self.pooling = pooling_shape
        self.im_argmax = None
        self.im_pool_shape = None
    
    def forward(self, input):
        im_num, h, w, d = input.shape
        p1, p2 = self.pooling

        self.output_pool = np.array(((h - p1)//p1 + 1, 
                        (w - p2)//p2 + 1, d), dtype=int)
        
        "Extracting the Maximum Value"
        im_pool = (viewW(input, (im_num, p1, p2, d)))[:,::p1,::p2].reshape(-1,
                                                    im_num, p1*p2, d).transpose(1,2,0,3)
        
        self.im_argmax = im_pool.argmax(axis=1)
        self.im_pool_shape = im_pool.shape
        
        im_max = im_pool.max(axis=1)

        return im_max.reshape(im_num, self.output_pool[0], self.output_pool[1], self.output_pool[2])
    
    def backward(self, input, grad_output):
        pack = []
        backprop = np.zeros(self.im_pool_shape)

        im_num, h, w, d = input.shape
        p1, p2 = self.pooling

        grad = grad_output.reshape(im_num, self.output_pool[0]*self.output_pool[1], self.output_pool[2])

        for i0 in range(im_num):
            for d0 in range(d):
                backprop[i0, self.im_argmax[i0,:,d0], np.arange(self.im_argmax.shape[1]), d0] = grad[i0,:,d0] #Assignment location
        
        reorder = backprop.reshape(im_num, p1, p2, self.output_pool[0], self.output_pool[1], self.output_pool[2])
        grad_input = reorder.transpose(0,3,1,4,2,5).reshape(im_num, h - (h%p1), w - (w%p2), d)
        return np.pad(grad_input, ((0,0), (0,h%p1), (0,w%p2), (0,0))), pack
    
    def get_output_shape(self, prev_output_shape):
        h, w, d = prev_output_shape
        p1, p2 = self.pooling
        return list(((h - p1)//p1 + 1, (w - p2)//p2 + 1, d))

    def get_weights(self):
        return []
    
    def set_weights(self, *weights):
        pass
    
class Conv2d():   
    def __init__(self, filters, kernel_shape, input_shape, padding = 'valid', learning_rate = 0.1, operation = 1, save_grad=False):
        self.learning_rate = learning_rate
        if padding == 'valid':
            output_shape = list((input_shape[0] - kernel_shape[0] + 1, input_shape[1] - kernel_shape[1] + 1, filters))
            kernel_dim = np.array(kernel_shape[0:2])-1
            pads = ((0,0),(0,0),(0,0),(0,0))
            back_pads = ((0,0),(kernel_dim[0],kernel_dim[0]), 
                          (kernel_dim[1], kernel_dim[1]), 
                          (0,0))
        elif padding == 'same':
            output_shape = list((input_shape[0], input_shape[1], filters))
            kernel_dim = np.array(kernel_shape[0:2])-1
            pads = ((0,0),(int(kernel_dim[0]/2), int(kernel_dim[0]/2) + int(kernel_dim[0]%2)),
                          (int(kernel_dim[1]/2), int(kernel_dim[1]/2) + int(kernel_dim[1]%2)),
                          (0,0))
            back_pads = ((0,0),(int(kernel_dim[0]/2) + int(kernel_dim[0]%2), int(kernel_dim[0]/2)),
                          (int(kernel_dim[1]/2) + int(kernel_dim[1]%2), int(kernel_dim[1]/2)),
                          (0,0))
        elif padding == 'full':
            output_shape = list((input_shape[0] + kernel_shape[0] - 1, input_shape[1] + kernel_shape[1] - 1, filters))
            kernel_dim = np.array(kernel_shape[0:2])-1
            pads = ((0,0),(kernel_dim[0],kernel_dim[0]), 
                          (kernel_dim[1], kernel_dim[1]), 
                          (0,0))
            back_pads = ((0,0),(0,0),(0,0),(0,0))
        else:
            print("No valid padding selected")
            exit()
        fan_in = np.array(input_shape).prod()
        fan_out = np.array(output_shape).prod()
        limit = np.sqrt(6 / (fan_in + fan_out))
        
        self.padding = padding
        self.pads = pads
        self.back_pads =  back_pads
        
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.filters = filters
        self.operation = operation
        self.grad_store_op = grad_store_op[save_grad]
        
        self.kernel = np.random.uniform(-limit, limit, size=(kernel_shape[0], kernel_shape[1], input_shape[2], filters))
        self.biases = np.zeros(self.filters)
    
    def forward(self, input):
        padded_input = np.pad(input, self.pads)
        resultant_shape = self.output_shape.copy()
        resultant_shape.insert(0, input.shape[0])
        
        # if(self.operation == 1):
        #     kernel_vec = self.kernel.reshape(-1, self.input_shape[2], self.output_shape[2]) #kernel[0]*kernel[1], channels, filters
            
        #     im_col = im2col_4d(padded_input, self.kernel.shape[0:3])        
        #     result = np.einsum('mijk, kjl -> mil', im_col, kernel_vec)
        #     result = result.reshape(resultant_shape)
    
        # else:
        result = conv2d_foward(padded_input, self.kernel, resultant_shape)
        return result + self.biases 
    
    def backward(self,input,grad_output):
        pack = []
        kernel_flip = np.flip(self.kernel.transpose(0,1,3,2),(0,1)) #Kernel is now (height, width, filters, depth) with heigth and depth rearranged so it is 180 deg flip
        padded_input = np.pad(input, self.pads)
        padded_grads = np.pad(grad_output, self.back_pads)
        grad = np.repeat(grad_output[:,:,:,np.newaxis] , input.shape[3], axis=3)
        
        # if(self.operation == 1):
        #     grad_vec = grad.reshape(grad.shape[0], -1, self.input_shape[2], self.output_shape[2])
        #     #------------------------------------------------------------------
            
        #     kernel_vec = kernel_flip.reshape(-1, self.output_shape[2], self.input_shape[2]) #(height*width, filters, depth)
        #     im_col_grad = im2col_4d(padded_grads, kernel_flip.shape[0:3])
        #     grad_input = np.einsum('mijk, kjl -> mil', im_col_grad, kernel_vec)
        #     pack.append(grad_input)
        #     pack.append(im_col_grad)
        #     pack.append(kernel_vec)
            
        #     grad_input = grad_input.reshape(input.shape)
            
        #     #------------------------------------------------------------------
        #     im_col = im2col_4d(padded_input, grad.shape[1:4])
        #     grad_weights = np.einsum('mijk, mkjl -> ijl', im_col, grad_vec)
        #     pack.append(grad_weights)
        #     pack.append(im_col)
        #     pack.append(grad_vec)
            
        #     grad_weights = grad_weights.reshape(self.kernel.shape)
        # else:
        grad_input = conv2d_foward(padded_grads, kernel_flip, input.shape)
        #------------------------------------------------------------------
        grad_weights = np.zeros(self.kernel.shape)
        self.grad_store_op(pack, grad_input, padded_grads, kernel_flip, input.shape)
        
        _, h0, w0, _ = padded_input.shape
        h1, w1, _ = self.output_shape
        for i0 in range(h0 - h1 + 1):
            for i1 in range(w0 - w1 + 1):
                grad_weights[i0, i1] = np.einsum('mijk, mijkl -> kl', padded_input[:, i0:(i0 + h1), i1:(i1 + w1)], grad)
        
        self.grad_store_op(pack, grad_weights, padded_input, grad, h0, h1, w0, w1)
        
        grad_biases = grad_output.mean(axis=0)*input.shape[0]
        
        # Here we perform a stochastic gradient descent step. 
        self.kernel = self.kernel - self.learning_rate * grad_weights
        self.biases = self.biases - self.learning_rate * grad_biases
        
        return grad_input, pack
    
    def set_weights(self, *weights):
        self.kernel = weights[0].copy()
        self.biases = weights[1].copy()
        
    def get_weights(self):
        return [self.kernel, self.biases]
    
    def get_output_shape(self, prev_output_shape=None):
        return self.output_shape

"""Extra Functions"""
def softmax_crossentropy_with_logits(logits,reference_answers):
    # Compute crossentropy from logits[batch,n_classes] and ids of correct answers
    #Reference_answers must be a 1-dimensional array, DO NOT USE HOT VECTORS
    logits_for_answers = logits[np.arange(len(logits)),reference_answers]
    
    xentropy = - logits_for_answers + np.log(np.sum(np.exp(logits),axis=-1))    
    return xentropy

def grad_softmax_crossentropy_with_logits(logits,reference_answers):
    # Compute crossentropy gradient from logits[batch,n_classes] and ids of correct answers
    #Reference_answers must be a 1-dimensional array, DO NOT USE HOT VECTORS
    ones_for_answers = np.zeros_like(logits)
    ones_for_answers[np.arange(len(logits)),reference_answers] = 1
    
    softmax = np.exp(logits) / np.exp(logits).sum(axis=-1,keepdims=True)
    return (- ones_for_answers + softmax) / logits.shape[0]

def sigmoid_binary_crossentroy_with_logits(logits, reference_answers):
    #Reference_answers must be a 2-dimensional array, DO NOT USE HOT VECTORS
    s = sig(logits)
    binary_entropy = -reference_answers*np.log(s) - (1-reference_answers)*np.log(1-s)
    return binary_entropy

def grad_sigmoid_binary_crossentroy_with_logits(logits, reference_answers):
    #Reference_answers must be a 2-dimensional array, DO NOT USE HOT VECTORS
    return -reference_answers + sig(logits)

def sig(z):
    return 1.0 / (1 + np.exp(-z))
   # return np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))

"Overall Foward Propagation"
def forward(network, X):
    # Compute activations of all network layers by applying them sequentially.
    # Return a list of activations for each layer. 
    
    activations = []
    input = X

    # Looping through each layer
    for l in network:
        activations.append(l.forward(input))
        # Updating input to last layer output
        input = activations[-1]
    
    assert len(activations) == len(network)
    return activations

"Function to Iterate through batches"
def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.random.permutation(len(inputs))
    t = trange(0, len(inputs) - batchsize + 1, batchsize)
    for start_idx in t:
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt], t

def iterate_minibatches_verbose_off(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.random.permutation(len(inputs))
    t = range(0, len(inputs) - batchsize + 1, batchsize)
    for start_idx in t:
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

def iterate_prediction(inputs, batchsize):
    t = range(0, len(inputs) - batchsize + 1, batchsize)
    for start_idx in t:
        excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt]

def iterate_loss(inputs, targets, batchsize):
    t = range(0, len(inputs) - batchsize + 1, batchsize)
    for start_idx in t:
        excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]