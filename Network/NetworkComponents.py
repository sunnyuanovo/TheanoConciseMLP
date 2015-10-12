import sys
sys.path.insert(0,'/u/yuanwei/scisoft/anaconda/lib/python2.7/site-packages')

sys.path.append('../Utilities')  
sys.path.append('/home/yw/workspace/test/TheanoConciseMLP/Utilities')  
import basic_utilities
import test_utilities


import random

import math
import numpy as np
import numpy
#import matplotlib.pyplot as plt
import theano
# By convention, the tensor submodule is loaded as T
import theano.tensor as T
import os
import os.path
#from main_rowbased import max_iteration

from theano import tensor

try:
    import cPickle as pickle
except ImportError:
    import pickle
    
functionmode = 'DebugMode'
functionmode = 'FAST_RUN'

class CosineLayer(object):
    """

    Given two inputs [q1 q2 .... qmb] and [d1 d2 ... dmb],
    this class computes the pairwise cosine similarity 
    between positive pairs and neg pairs
    """
    def output_noloop_1(self, index_Q, index_D, Q, D):
        """
        This function do the same as previous function "output", except that no loop or scan is used here
        In this version 1, we don't use any broadcasting.
        
        Input: index_Q is T.ivector(), shape = (?,)
        Input: index_D is T.ivector(), shape = (?,)
        Input: Q is T.fmatrix(), shape=(bs, eb)
        Input: D is T.fmatrix(), shape = (bs, eb)
        Output: First, get two new matrices Q[index_Q] and D[index_D], both with shape (batch_size*(neg+1), embed_size).
                Then for each row vector pair, compute cosine value
        """
        Q_view = Q[index_Q]
        D_view = D[index_D]
        
        dotQD = (Q_view * D_view).sum(axis = 1) #  Q[inds_Q]*D[inds_D]
        dotQQ = (Q_view * Q_view).sum(axis = 1) #  Q[inds_Q]*D[inds_D]
        dotDD = (D_view * D_view).sum(axis = 1) #  Q[inds_Q]*D[inds_D]
        
        dotQQDD_sqrt = tensor.sqrt(dotQQ*dotDD)
        
        return dotQD/dotQQDD_sqrt
        
    def __init__(self, n_mbsize, n_neg, n_shift):
        """ Initialize the parameters of the logistic regression

        :type inputQ and inputD: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)
        
        :type n_neg: int
        :param n_neg: number of negative samples
        
        :type n_shift: int
        :param n_out: shift of negative samples

        """
        # keep track of model input and target.
        # We store a flattened (vector) version of target as y, which is easier to handle
        self.n_mbsize = n_mbsize
        self.n_neg = n_neg
        self.n_shift = n_shift

class Minibatch(object):
    def __init__(self, SegSize, ElementSize, SampleIdx, SegIdx, FeaIdx, FeaVal):
        '''
        An instance of a single minibatch
        All parameters are with the same names as those in WordHash

        '''
        
        self.SegSize = SegSize # int scalar
        self.ElementSize = ElementSize # int scalar
        
        self.m_rgSampleIdx = SampleIdx # an array of int
        self.m_rgSegIdx = SegIdx # an array of int
        self.m_rgFeaIdx = FeaIdx # an array of int
        self.m_rgFeaVal = FeaVal # an array of float

class InputStream(object):
    # 1. Load and set last 5 integers
    # 2. Load in all minibatches
    def __init__(self, filename):
        f = open(filename, "rb")
        f.seek(-20, 2)
        c = np.fromfile(f, dtype=np.uint32)

        self.nMaxFeatureId =  c[0] # nMaxFeatureId
        self.nLine = c[1] # nLine
        self.nMaxSegmentSize = c[2] #  nMaxSegmentSize
        self.nMaxFeatureNum = c[3] # nMaxFeatureNum
        self.BatchSize = c[4] # BatchSize
        
        if self.nLine % self.BatchSize == 0:
            self.nTotalBatches = self.nLine / self.BatchSize
        else:
            self.nTotalBatches = self.nLine / self.BatchSize + 1
        
        self.minibatches = [] # initial with an empty list
        f.seek(0) # move to the beginning
        for i in range(self.nTotalBatches):
            SegSize, ElementSize = np.fromfile(f, dtype=np.uint32, count=2)
            SampleIdx = np.fromfile(f, dtype=np.uint32, count=SegSize)
            SegIdx = np.fromfile(f, dtype=np.uint32, count=SegSize)
            FeaIdx = np.fromfile(f, dtype=np.uint32, count=ElementSize)
            FeaVal = np.fromfile(f, dtype=np.float32, count=ElementSize)
            self.minibatches.append(Minibatch(SegSize, ElementSize, SampleIdx, SegIdx, FeaIdx, FeaVal))
        f.close()
        
    def setaminibatch(self, curr_minibatch, i):
        '''
        Set up current minibatch using self.minibatches[i]
        minibatch is of Shape (inputstream1.BatchSize, inputstream1.nMaxFeatureId)
        if this is the last batch, reshape it if necessary
        '''
        assert(i >= 0 and i < len(self.minibatches))
        
        curr_minibatch.fill(0.0)
        minibatch_cols = curr_minibatch.shape[1] # used for remove OOV
        
#        tmp_minibatch = self.minibatches[i]
        segid = 0 # default value
        
        for j in range(self.minibatches[i].SegSize):
            # Suppose the array is [2,5,7]
            # Iter 1: segid = 0, segidx = 2, this means we are working on the 1st query (seg), its featureid from [0,2)
            # Iter 2: segid = 1, segidx = 5, this means we are working on the 2nd query (seg), its featureid from [2,5)
            # Iter 3: segid = 2, segidx = 7, this means we are working on the 3rd query (seg), its featureid from [5,7)
            segidx = self.minibatches[i].m_rgSegIdx[j]
            if j == 0:
                prev_segidx = 0
            else:
                prev_segidx = self.minibatches[i].m_rgSegIdx[j-1]
            

            for k in range(prev_segidx, segidx):
                feaid = self.minibatches[i].m_rgFeaIdx[k]
                feaval = self.minibatches[i].m_rgFeaVal[k]

                # This step is to remove OOVs
                # The shape of curr_minibatch is fixed. any index out of it is OOV                
                if feaid < minibatch_cols:
                    curr_minibatch[segid, feaid] = feaval

            # empty check
            if curr_minibatch[segid, :].sum() == 0.0:
                curr_minibatch[segid, -1] = 1.0 
 
            
            segid = segid +1
            
class LayerWithoutBias(object):
    def __init__(self, W_init, activation):
        '''
        A layer of a neural network, computes s(Wx) where s is a nonlinearity and x is the input vector.

        :parameters:
            - W_init : np.ndarray, shape=(n_output, n_input)
                Values to initialize the weight matrix to.
            - activation : theano.tensor.elemwise.Elemwise
                Activation function for layer output
        '''
        # Retrieve the input and output dimensionality based on W's initialization
#        n_input, n_output = W_init.shape
        # All parameters should be shared variables.
        # They're used in this class to compute the layer output,
        # but are updated elsewhere when optimizing the network parameters.
        # Note that we are explicitly requiring that W_init has the theano.config.floatX dtype
        self.W = theano.shared(value=W_init.astype(theano.config.floatX),
                               # The name parameter is solely for printing purporses
                               name='W',
                               # Setting borrow=True allows Theano to use user memory for this object.
                               # It can make code slightly faster by avoiding a deep copy on construction.
                               # For more details, see
                               # http://deeplearning.net/software/theano/tutorial/aliasing.html
                               borrow=True)
        self.activation = activation
        # We'll compute the gradient of the cost of the network with respect to the parameters in this list.
        self.params = [self.W]
        
    def output(self, x):
        '''
        Compute this layer's output given an input
        
        :parameters:
            - x : theano.tensor.var.TensorVariable
                Theano symbolic variable for layer input

        :returns:
            - output : theano.tensor.var.TensorVariable
                Mixed, biased, and activated x
        '''
        # Compute linear mix
        lin_output = T.dot(x, self.W)
        # Output is just linear mix if no activation function
        # Otherwise, apply the activation function
        return (lin_output if self.activation is None else self.activation(lin_output))
    def output_linear(self, x):
        '''
        Compute this layer's output given an input
        
        :parameters:
            - x : theano.tensor.var.TensorVariable
                Theano symbolic variable for layer input

        :returns:
            - output : theano.tensor.var.TensorVariable
                Mixed, biased, and activated x
        '''
        # Compute linear mix
        lin_output = T.dot(x, self.W)
        return lin_output

