import sys
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

try:
    import cPickle as pickle
except ImportError:
    import pickle
    
functionmode = 'DebugMode'
functionmode = 'FAST_RUN'

class SimpleDSSMModelFormat(object):
    # only mlayer_num and mlink_num are integers
    # all other parameters are ndarray
    def __init__(self, mlayer_num, layer_info, mlink_num, in_num_list, out_num_list, params):
        self.mlayer_num = mlayer_num
        self.layer_info = layer_info
        self.mlink_num = mlink_num
        self.in_num_list = in_num_list
        self.out_num_list = out_num_list
        self.params = params
        
    
def generate_index(n_mbsize, n_neg, n_shift):
        # Next, we need to generate 2 lists of index
        # these 2 lists together have mbsize*(neg+1) element
        # after reshape, it should be (mbsize, neg+1) matrix
        index_train_Q = numpy.arange(0)
        index_train_D = numpy.arange(0)
        
        for tmp_index in range(n_mbsize):
            # for current sample, it's positive pair is itself
            index_train_Q = numpy.append(index_train_Q, [tmp_index] * (n_neg+1))
            index_train_D = numpy.append(index_train_D, [tmp_index])
            index_train_D = numpy.append(index_train_D, [(tmp_index+n_shift+y)%n_mbsize for y in range(n_neg)])
            
#            index_train_D += [tmp_index]
#            index_train_D += [(tmp_index+n_shift+y)%n_mbsize for y in range(n_neg)]
        
        index_test_Q = numpy.arange(n_mbsize)
        index_test_D = numpy.arange(n_mbsize)
#        theano.tensor.as_tensor_variable(x, name=None, ndim=None)

#        indexes = [ theano.shared(index_train_Q), theano.shared(index_train_Q), theano.shared(index_test_Q), theano.shared(index_test_D)]
        indexes = [ index_train_Q.astype('int32'), index_train_D.astype('int32'), index_test_Q.astype('int32'), index_test_D.astype('int32')]
        return indexes

class CosineLayer(object):
    """

    Given two inputs [q1 q2 .... qmb] and [d1 d2 ... dmb],
    this class computes the pairwise cosine similarity 
    between positive pairs and neg pairs
    """
    def ComputeCosineBetweenTwoVectors(self, q_ind, d_ind, Q, D):
        '''
        Compute Cosine similarity between two vectors: Q[q_ind] and D[d_ind]
        
        :parameters:
            - q_ind, d_ind : int, i.e. Theano scalars
                two indexes for corresponding vectors

            - Q,D : theano.tensor.var.TensorVariable
                Theano symbolic variable for layer input

        :returns:
            - output : theano.tensor.var.TensorVariable
                Cosine value between two vectors
        '''

        # index is like (1,2)
        q = Q[q_ind]
        d = D[d_ind]
        qddot = T.dot(q,d)
        q_norm = T.sqrt((q**2).sum())
        d_norm = T.sqrt((d**2).sum())
        return qddot/(q_norm * d_norm)

    # for train, we need to compute a cosine matrix for (Q,D), then compute a final score
    def forward_train(self, index_Q, index_D, Q, D):   
        '''
        Compute Cosine Matrix based on Q and D
\        
        :parameters:
            - index_Q, index_D : each is a list of integers
                two indexes for corresponding vectors

            - Q,D : theano.tensor.var.TensorVariable
                Theano symbolic variable for layer input

        :returns:
            - output : theano.tensor.var.TensorVariable
                Sum of log likelihood probabilities
        '''
        
        components, updates = theano.scan(self.ComputeCosineBetweenTwoVectors,
                                  outputs_info=None,
                                  sequences=[index_Q, index_D],
                                  non_sequences=[Q,D])
        
        components_reshape = T.reshape(components, (self.n_mbsize, self.n_neg+1))
        
        # for this matrix, each line is a prob distribution right now.
        components_reshape_softmax = T.nnet.softmax(components_reshape)
        
        # get the first column
        column1 = components_reshape_softmax[:,0]
        
        # get the final output
        return  (-1 * column1.sum())

    # for test, we only need to compute a cosine vector for (Q,D)
    def forward_test(self, index_Q, index_D, Q, D):   
        # components is a vector         
        components, updates = theano.scan(self.ComputeCosineBetweenTwoVectors,
                                  outputs_info=None,
                                  sequences=[index_Q, index_D],
                                  non_sequences=[Q,D])
        
        
        # get the final output
        return components
    # for test, we only need to compute a cosine vector for (Q,D)
    # the output result is just a list. the number is the same as those in index_Q/D
    # how to use the result is described in Class DSSM
    def output(self, index_Q, index_D, Q, D):   
        # components is a vector         
        components, updates = theano.scan(self.ComputeCosineBetweenTwoVectors,
                                  outputs_info=None,
                                  sequences=[index_Q, index_D],
                                  non_sequences=[Q,D])
        
        
        # get the final output
        return components
        
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
        
    def backup__init__(self, c):
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
        
    def display(self):
        print self.nMaxFeatureId, self.nLine, self.nMaxSegmentSize, self.nMaxFeatureNum, self.BatchSize, self.nTotalBatches
        for item in self.minibatches:
            print item.SegSize, item.ElementSize, item.m_rgSampleIdx, item.m_rgSegIdx, item.m_rgFeaIdx, item.m_rgFeaVal
            print "\n"# = SegSize # int scalar
 
    def loadinoneminibatch(self, SegSize, ElementSize, SampleIdx, SegIdx, FeaIdx, FeaVal):
        self.minibatches.append(Minibatch(SegSize, ElementSize, SampleIdx, SegIdx, FeaIdx, FeaVal))
        
    def loadinallminibatches(self, f):
        f.seek(0) # move to the beginning
        for i in range(self.nTotalBatches):
            SegSize, ElementSize = np.fromfile(f, dtype=np.uint32, count=2)
            SampleIdx = np.fromfile(f, dtype=np.uint32, count=SegSize)
            SegIdx = np.fromfile(f, dtype=np.uint32, count=SegSize)
            FeaIdx = np.fromfile(f, dtype=np.uint32, count=ElementSize)
            FeaVal = np.fromfile(f, dtype=np.float32, count=ElementSize)
            self.minibatches.append(Minibatch(SegSize, ElementSize, SampleIdx, SegIdx, FeaIdx, FeaVal))
            
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
            
            segid = segid +1
            
        
               

class Layer(object):
    def __init__(self, W_init, b_init, activation):
        '''
        A layer of a neural network, computes s(Wx + b) where s is a nonlinearity and x is the input vector.

        :parameters:
            - W_init : np.ndarray, shape=(n_output, n_input)
                Values to initialize the weight matrix to.
            - b_init : np.ndarray, shape=(n_output,)
                Values to initialize the bias vector
            - activation : theano.tensor.elemwise.Elemwise
                Activation function for layer output
        '''
        # Retrieve the input and output dimensionality based on W's initialization
        n_input, n_output = W_init.shape
        # Make sure b is n_output in size
        assert b_init.shape == (n_output,)
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
        # We can force our bias vector b to be a column vector using numpy's reshape method.
        # When b is a column vector, we can pass a matrix-shaped input to the layer
        # and get a matrix-shaped output, thanks to broadcasting (described below)
        self.b = theano.shared(value=b_init.reshape(1, n_output).astype(theano.config.floatX),
                               name='b',
                               borrow=True,
                               # Theano allows for broadcasting, similar to numpy.
                               # However, you need to explicitly denote which axes can be broadcasted.
                               # By setting broadcastable=(False, True), we are denoting that b
                               # can be broadcast (copied) along its second dimension in order to be
                               # added to another variable.  For more information, see
                               # http://deeplearning.net/software/theano/library/tensor/basic.html
                               broadcastable=(True, False))
        self.activation = activation
        # We'll compute the gradient of the cost of the network with respect to the parameters in this list.
        self.params = [self.W, self.b]
        
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
        lin_output = T.dot(x, self.W) + self.b
        # Output is just linear mix if no activation function
        # Otherwise, apply the activation function
        return (lin_output if self.activation is None else self.activation(lin_output))
    
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
        n_input, n_output = W_init.shape
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

class MLP(object):
    def __init__(self, W_init, b_init, activations):
        '''
        Multi-layer perceptron class, computes the composition of a sequence of Layers

        :parameters:
            - W_init : list of np.ndarray, len=N
                Values to initialize the weight matrix in each layer to.
                The layer sizes will be inferred from the shape of each matrix in W_init
            - b_init : list of np.ndarray, len=N
                Values to initialize the bias vector in each layer to
            - activations : list of theano.tensor.elemwise.Elemwise, len=N
                Activation function for layer output for each layer
        '''
        # Make sure the input lists are all of the same length
        assert len(W_init) == len(b_init) == len(activations)
        
        # Initialize lists of layers
        self.layers = []
        # Construct the layers
        for W, b, activation in zip(W_init, b_init, activations):
            self.layers.append(Layer(W, b, activation))

        # Combine parameters from all layers
        self.params = []
        for layer in self.layers:
            self.params += layer.params
        
    def output(self,  x):
        '''
        Compute the MLP's output given an input
        
        :parameters:
            - x : theano.tensor.var.TensorVariable
                Theano symbolic variable for network input

        :returns:
            - output : theano.tensor.var.TensorVariable
                x passed through the MLP
        '''
        # Recursively compute output
        for layer in self.layers:
            x = layer.output(x)
        return x

    def squared_error(self, x, y):
        '''
        Compute the squared euclidean error of the network output against the "true" output y
        
        :parameters:
            - x : theano.tensor.var.TensorVariable
                Theano symbolic variable for network input
            - y : theano.tensor.var.TensorVariable
                Theano symbolic variable for desired network output

        :returns:
            - error : theano.tensor.var.TensorVariable
                The squared Euclidian distance between the network output and y
        '''
        return T.sum((self.output(x) - y)**2)

class DSSM(object):
    def __init__(self, W_init_Q, activations_Q, W_init_D, activations_D, n_mbsize, n_neg, n_shift, strategy = 0):
        '''
        This class is similar to MLP, except that we need to construct separate models for Q and D, 
        then add a cosine label at the end
        
        :parameters:
            - W_init_Q/D : list of np.ndarray
                Values to initialize the weight matrix in each layer to.
                The layer sizes will be inferred from the shape of each matrix in W_init
            - activations_Q/D : list of theano.tensor.elemwise.Elemwise
                Activation function for layer output for each layer
        '''
        
        if strategy == 0:
            # Make sure the input lists are all of the same length
            assert len(W_init_Q)  == len(activations_Q)
            assert len(W_init_D)  == len(activations_D)
            
            # Initialize lists of layers
            self.layers_Q = []
            for W, activation in zip(W_init_Q, activations_Q):
                self.layers_Q.append(LayerWithoutBias(W, activation))
            
            self.layers_D = []
            for W, activation in zip(W_init_D, activations_D):
                self.layers_D.append(LayerWithoutBias(W, activation))
                
            self.layer_cosine = CosineLayer(n_mbsize, n_neg, n_shift)
    
            # Combine parameters from all layers
            self.params = []
            self.params_Q = []
            self.params_D = []
            
            for layer in self.layers_Q:
                self.params += layer.params
                self.params_Q += layer.params
            for layer in self.layers_D:
                self.params += layer.params
                self.params_D += layer.params
 

    # This function is for backup
    def backup__init__backup(self, W_init, n_mbsize, n_neg, n_shift,  activations, strategy = 0):
        '''
        This class is similar to MLP, except that we need to construct separate models for Q and D, 
        then add a cosine label at the end
        
        :parameters:
            - W_init : list of np.ndarray, len=N
                Values to initialize the weight matrix in each layer to.
                The layer sizes will be inferred from the shape of each matrix in W_init
            - activations : list of theano.tensor.elemwise.Elemwise, len=N
                Activation function for layer output for each layer
        '''
        
        if strategy == 0:
            # Make sure the input lists are all of the same length
            assert len(W_init)  == len(activations)
            
            # Initialize lists of layers
            self.layers_Q = []
            self.layers_D = []
            # Construct the layers
            for W, activation in zip(W_init, activations):
                self.layers_Q.append(LayerWithoutBias(W, activation))
                self.layers_D.append(LayerWithoutBias(W, activation))
            
            self.layer_cosine = CosineLayer(n_mbsize, n_neg, n_shift)
    
            # Combine parameters from all layers
            self.params = []
            for layer in self.layers_Q:
                self.params += layer.params
            for layer in self.layers_D:
                self.params += layer.params
        elif strategy == 1:
            # Make sure the input lists are all of the same length
            # In this case, the W_init is a list of trained weights, first half is from Q, 2nd half is from D
            assert len(W_init)  == len(activations)*2
            
            # Initialize lists of layers
            self.layers_Q = []
            self.layers_D = []
            halflen = len(activations)
            
            for W, activation in zip(W_init[0:halflen], activations):
                self.layers_Q.append(LayerWithoutBias(W, activation))
                
            for W, activation in zip(W_init[halflen:], activations):
                self.layers_D.append(LayerWithoutBias(W, activation))
            
            self.layer_cosine = CosineLayer(n_mbsize, n_neg, n_shift)
    
            # Combine parameters from all layers
            self.params = []
            for layer in self.layers_Q:
                self.params += layer.params
            for layer in self.layers_D:
                self.params += layer.params
            
            
    def output_train(self, index_Q, index_D, Q, D):
        '''
        Compute the DSSM's output given an input
        
        :parameters:
            - index_Q, index_D : each is a list of integers, i.e. two tensor vectors
                two indexes for corresponding vectors

            - Q,D : theano.tensor.var.TensorVariable, should be two matrices
                Theano symbolic variable for layer input

        :returns:
            - output : theano.tensor.var.TensorVariable, should be a tensor matrix
                A scalar value
        '''
        # Recursively compute output
        for layer in self.layers_Q:
            Q = layer.output(Q)
        for layer in self.layers_D:
            D = layer.output(D)
        
        cosine_matrix = self.layer_cosine.output(index_Q, index_D, Q, D)
        cosine_matrix_reshape = T.reshape(cosine_matrix, (self.layer_cosine.n_mbsize, self.layer_cosine.n_neg+1))
        
        # for this matrix, each line is a prob distribution right now.
#        cosine_matrix_reshape_softmax = T.nnet.softmax(cosine_matrix_reshape)
        
        # get the first column
        column1 = cosine_matrix_reshape[:,0]
        
        # get the final output
        return  (-1 * column1.sum())
    def output_train_test(self, index_Q, index_D, Q, D):
        '''
        Compute the DSSM's output given an input
        
        :parameters:
            - index_Q, index_D : each is a list of integers, i.e. two tensor vectors
                two indexes for corresponding vectors

            - Q,D : theano.tensor.var.TensorVariable, should be two matrices
                Theano symbolic variable for layer input

        :returns:
            - output : theano.tensor.var.TensorVariable, should be a tensor matrix
                A scalar value
        '''
        # Recursively compute output
        for layer in self.layers_Q:
            Q = layer.output(Q)
        for layer in self.layers_D:
            D = layer.output(D)
#        return Q, D
        
        cosine_matrix = self.layer_cosine.output(index_Q, index_D, Q, D) * 10
    
        cosine_matrix_reshape = T.reshape(cosine_matrix, (self.layer_cosine.n_mbsize, self.layer_cosine.n_neg+1))
        
        # for this matrix, each line is a prob distribution right now.
        cosine_matrix_reshape_softmax = T.nnet.softmax(cosine_matrix_reshape)
#        return cosine_matrix_reshape_softmax
        
        # get the first column
        column1 = cosine_matrix_reshape_softmax[:,0]
        
        column1_neglog = -T.log(column1)
        return column1_neglog.sum()
        
        # get the final output
#        return  (-1 * column1.sum())
    def output_test(self, index_Q, index_D, Q, D):
        '''
        Compute the DSSM's output given an input
        
        :parameters:
            - index_Q, index_D : each is a list of integers, i.e. two tensor vectors
                two indexes for corresponding vectors

            - Q,D : theano.tensor.var.TensorVariable, should be two matrices
                Theano symbolic variable for layer input

        :returns:
            - output : theano.tensor.var.TensorVariable, should be a tensor matrix
                A tensor matrix
        '''
        # Recursively compute output
        for layer in self.layers_Q:
            Q = layer.output(Q)
        for layer in self.layers_D:
            D = layer.output(D)
        
        cosine_matrix = self.layer_cosine.output(index_Q, index_D, Q, D)
        cosine_matrix_reshape = T.reshape(cosine_matrix, (self.layer_cosine.n_mbsize, 1))
        
        return cosine_matrix_reshape


def gradient_updates_momentum(cost, params, learning_rate, momentum):
    '''
    Compute updates for gradient descent with momentum
    
    :parameters:
        - cost : theano.tensor.var.TensorVariable
            Theano cost function to minimize
        - params : list of theano.tensor.var.TensorVariable
            Parameters to compute gradient against
        - learning_rate : float
            Gradient descent learning rate
        - momentum : float
            Momentum parameter, should be at least 0 (standard gradient descent) and less than 1
   
    :returns:
        updates : list
            List of updates, one for each parameter
    '''
    # Make sure momentum is a sane value
    assert momentum < 1 and momentum >= 0
    # List of update steps for each parameter
    updates = []
    # Just gradient descent on cost
    for param in params:
        """
        # For each parameter, we'll create a param_update shared variable.
        # This variable will keep track of the parameter's update step across iterations.
        # We initialize it to 0
        param_update = theano.shared(param.get_value()*0., broadcastable=param.broadcastable)
        # Each parameter is updated by taking a step in the direction of the gradient.
        # However, we also "mix in" the previous step according to the given momentum value.
        # Note that when updating param_update, we are using its old value and also the new gradient step.
        updates.append((param, param - learning_rate*param_update))
        # Note that we don't need to derive backpropagation to compute updates - just use T.grad!
        updates.append((param_update, momentum*param_update + (1. - momentum)*T.grad(cost, param)))
        """
#        param_update = theano.shared(param.get_value()*0., broadcastable=param.broadcastable)
#        updates.append((param_update, T.grad(cost, param)))
        updates.append((param, param - learning_rate*T.grad(cost, param)))
    return updates

# We'll only train the network with 20 iterations.
# A more common technique is to use a hold-out validation set.
# When the validation error starts to increase, the network is overfitting,
# so we stop training the net.  This is called "early stopping", which we won't do here.
def test_sim():
    # Training data - two randomly-generated Gaussian-distributed clouds of points in 2d space
    np.random.seed(0)
    # Number of points
    N = 10
    # Labels for each cluster
    y = np.random.random_integers(0, 1, N)
    # Mean of each cluster
    means = np.array([[-1, 1], [-1, 1]])
    # Covariance (in X and Y direction) of each cluster
    covariances = (np.random.random_sample((2, 2)) + 1).astype(np.float32)
    
    #print type(covariances), covariances.dtype
    # Dimensions of each point
    X = np.vstack([np.random.randn(N)*covariances[0, y] + means[0, y],
                   np.random.randn(N)*covariances[1, y] + means[1, y]])
    X = X.astype(np.float32)
    y = np.float32(y)
    X = np.transpose(X)
    y = np.transpose(y)
    y = np.reshape(y, (N,1))
    print type(X), X.shape, type(y), y.shape
    
    
    # First, set the size of each layer (and the number of layers)
    # Input layer size is training data dimensionality (2)
    # Output size is just 1-d: class label - 0 or 1
    # Finally, let the hidden layers be twice the size of the input.
    # If we wanted more layers, we could just add another layer size to this list.
    layer_sizes = [X.shape[1], X.shape[1]*2, 1]
    print layer_sizes
    # Set initial parameter values
    W_init = []
    b_init = []
    activations = []
    for n_input, n_output in zip(layer_sizes[:-1], layer_sizes[1:]):
        print "n_input,n_output = %d,%d" % (n_input, n_output)
        # Getting the correct initialization matters a lot for non-toy problems.
        # However, here we can just use the following initialization with success:
        # Normally distribute initial weights
        W_init.append(np.random.randn(n_input, n_output).astype(np.float32))
    #    print W_init[-1].dtype
        # Set initial biases to 1
        b_init.append(np.ones(n_output).astype(np.float32))
        # We'll use sigmoid activation for all layers
        # Note that this doesn't make a ton of sense when using squared distance
        # because the sigmoid function is bounded on [0, 1].
        activations.append(T.nnet.sigmoid)
    # Create an instance of the MLP class
    mlp = MLP(W_init, b_init, activations)
    
    # Create Theano variables for the MLP input
    mlp_input = T.matrix('mlp_input')
    # ... and the desired output
    mlp_target = T.col('mlp_target')
    # Learning rate and momentum hyperparameter values
    # Again, for non-toy problems these values can make a big difference
    # as to whether the network (quickly) converges on a good local minimum.
    learning_rate = 0.01
    momentum = 0.9
    # Create a function for computing the cost of the network given an input
    cost = mlp.squared_error(mlp_input, mlp_target)
    # Create a theano function for training the network
    train = theano.function([mlp_input, mlp_target], cost,
                            updates=gradient_updates_momentum(cost, mlp.params, learning_rate, momentum), mode=functionmode)
    # Create a theano function for computing the MLP's output given some input
    mlp_output = theano.function([mlp_input], mlp.output(mlp_input), mode=functionmode)
    
    # Keep track of the number of training iterations performed
    iteration = 0
    
    
    
    max_iteration = 5
    while iteration < max_iteration:
        # Train the network using the entire training set.
        # With large datasets, it's much more common to use stochastic or mini-batch gradient descent
        # where only a subset (or a single point) of the training set is used at each iteration.
        # This can also help the network to avoid local minima.
        
        
        current_cost = train(X, y)
        # Get the current network output for all points in the training set
        current_output = mlp_output(X)
        
        
        
        # We can compute the accuracy by thresholding the output
        # and computing the proportion of points whose class match the ground truth class.
        accuracy = np.mean((current_output > .5) == y)
        print iteration, accuracy
        iteration += 1


def test_dssm():
    # Training data - two randomly-generated Gaussian-distributed clouds of points in 2d space
    np.random.seed(0)
    # Number of points
    N = 4
    # Labels for each cluster
    y = np.random.random_integers(0, 1, N)
    # Mean of each cluster
    means = np.array([[-1, 1], [-1, 1]])
    # Covariance (in X and Y direction) of each cluster
    covariances = (np.random.random_sample((2, 2)) + 1).astype(np.float32)
    
    #print type(covariances), covariances.dtype
    # Dimensions of each point
    X = np.vstack([np.random.randn(N)*covariances[0, y] + means[0, y],
                   np.random.randn(N)*covariances[1, y] + means[1, y]])
    X = X.astype(np.float32)
    X = np.transpose(X)

    print type(X), X.shape, X.dtype
    print "X is as follows"
    print X
    
    X1 = np.vstack([np.random.randn(N)*covariances[0, y] + means[0, y],
                   np.random.randn(N)*covariances[1, y] + means[1, y]])
    X1 = X1.astype(np.float32)
    X1 = np.transpose(X1)
    print "X1 is as follows"
    print X1
    
    
    # First, set the size of each layer (and the number of layers)
    # Input layer size is training data dimensionality (2)
    # Output size is just 1-d: class label - 0 or 1
    # Finally, let the hidden layers be twice the size of the input.
    # If we wanted more layers, we could just add another layer size to this list.
    layer_sizes = [X.shape[1], X.shape[1]*2]#, X.shape[1]*2]#, X.shape[1]*2, X.shape[1]*2]
    print "layer_sizes is as follows:"
    print layer_sizes
    # Set initial parameter values
    W_init = []
    b_init = []
    activations = []
    for n_input, n_output in zip(layer_sizes[:-1], layer_sizes[1:]):
        print "n_input,n_output = %d,%d" % (n_input, n_output)
        # Getting the correct initialization matters a lot for non-toy problems.
        # However, here we can just use the following initialization with success:
        # Normally distribute initial weights
        W_init.append(np.random.randn(n_input, n_output).astype(np.float32))
    #    print W_init[-1].dtype
        # Set initial biases to 1
        b_init.append(np.ones(n_output).astype(np.float32))
        # We'll use sigmoid activation for all layers
        # Note that this doesn't make a ton of sense when using squared distance
        # because the sigmoid function is bounded on [0, 1].
#        activations.append(T.nnet.sigmoid)
        activations.append(T.tanh)
    # Create an instance of the MLP class
    mbsize = 4
    neg = 2
    shift = 2
    indexes = generate_index(mbsize, neg, shift)
    print indexes
    print indexes[0].dtype
    
    dssm = DSSM(W_init, b_init, mbsize, neg, shift, activations)
    
    print "W_init is as follows:"
    print W_init
    
    print "b_init is as follows:"
    print b_init
    
    # Create Theano variables for the MLP input
    dssm_index_Q = T.ivector('dssm_index_Q')
    dssm_index_D = T.ivector('dssm_index_D')
    dssm_input_Q = T.matrix('dssm_input_Q')
    dssm_input_D = T.matrix('dssm_input_D')
    # ... and the desired output
#    mlp_target = T.col('mlp_target')
    # Learning rate and momentum hyperparameter values
    # Again, for non-toy problems these values can make a big difference
    # as to whether the network (quickly) converges on a good local minimum.
    learning_rate = 0.01
    momentum = 0.9
    # Create a function for computing the cost of the network given an input
#    cost = mlp.squared_error(mlp_input, mlp_target)
    cost = dssm.output_train(dssm_index_Q, dssm_index_D, dssm_input_Q, dssm_input_D)
    cost_test = dssm.output_test(dssm_index_Q, dssm_index_D, dssm_input_Q, dssm_input_D)
        
    # Create a theano function for training the network
    train = theano.function([dssm_index_Q, dssm_index_D, dssm_input_Q, dssm_input_D], cost,
                            updates=gradient_updates_momentum(cost, dssm.params, learning_rate, momentum), mode=functionmode)
    # Create a theano function for computing the MLP's output given some input
    dssm_output = theano.function([dssm_index_Q, dssm_index_D, dssm_input_Q, dssm_input_D], cost_test, mode=functionmode)
    
    
    ywcost = dssm.output_train_test(dssm_index_Q, dssm_index_D, dssm_input_Q, dssm_input_D)
    ywtest = theano.function([dssm_index_Q, dssm_index_D, dssm_input_Q, dssm_input_D], ywcost,
                             updates=gradient_updates_momentum(ywcost, dssm.params, learning_rate, momentum), mode=functionmode)
    
    # Keep track of the number of training iterations performed
    iteration = 0
    max_iteration = 3
    while iteration < max_iteration:
        # Train the network using the entire training set.
        # With large datasets, it's much more common to use stochastic or mini-batch gradient descent
        # where only a subset (or a single point) of the training set is used at each iteration.
        # This can also help the network to avoid local minima.
        
        
#        current_cost = train(X, y)
        # Get the current network output for all points in the training set
#        current_output = mlp_output(X)

        print "Iteration %d--------------" % (iteration)
        
        
#        current_cost = train(indexes[0], indexes[1], X, X)
#        print iteration, current_cost
#        current_output = dssm_output(indexes[2], indexes[3], X, X)
#        current_output = dssm_output(indexes[2], indexes[3], X, X1)
#        print "indexes[0] = ", indexes[0]
#        print "indexes[1] = ", indexes[1]
        
        current_output = ywtest(indexes[0], indexes[1], X, X1)
#        current_output = ywtest(X, X1)
        print current_output
  
              
        
        
        
        # We can compute the accuracy by thresholding the output
        # and computing the proportion of points whose class match the ground truth class.
        
        iteration += 1
        

def train_dssm_with_minibatch(bin_file_train_1, bin_file_train_2, dssm_file_1_simple, dssm_file_2_simple, outputdir, ntrial, shift, max_iteration):
    # 1. Load in the input streams
    # Suppose the max seen feaid in the stream is 48930
    # then, inputstream1.nMaxFeatureId is 48931, which is one more
    # Ideally, the num_cols should be 48931
    # However, to make it conpatible with MS DSSM toolkit, we add it by one
    inputstream1 = InputStream(bin_file_train_1) # this will load in the whole file as origin. No modification at all
    inputstream2 = InputStream(bin_file_train_2)
    

    # 2. Load in the network structure and initial weights from DSSM
    init_model_1 = load_simpledssmmodel(dssm_file_1_simple)
    activations_1 = [T.tanh] * init_model_1.mlink_num
    
    init_model_2 = load_simpledssmmodel(dssm_file_2_simple)
    activations_2 = [T.tanh] * init_model_2.mlink_num

    # Before iteration, dump out the init model 
    outfilename_1 = os.path.join(outputdir, "yw_dssm_Q_0")
    outfilename_2 = os.path.join(outputdir, "yw_dssm_D_0")
    save_simpledssmmodel(outfilename_1, init_model_1)
    save_simpledssmmodel(outfilename_2, init_model_2)
    


    
    # 3. Generate useful index structures
    # We assue that each minibatch is of the same size, i.e. mbsize
    # if the last batch has fewer samples, just ignore it
    mbsize = inputstream1.BatchSize
    indexes = generate_index(mbsize, ntrial, shift) # for a normal minibatch, we should use this indexes
#    indexes_lastone = generate_index(inputstream1.minibatches[-1].SegSize, ntrial, shift) # this is used for the last batch

    # 4. Generate an instance of DSSM    
    dssm = DSSM(init_model_1.params, activations_1, init_model_2.params, activations_2, mbsize, ntrial, shift )

    # Create Theano variables for the MLP input
    dssm_index_Q = T.ivector('dssm_index_Q')
    dssm_index_D = T.ivector('dssm_index_D')
    dssm_input_Q = T.matrix('dssm_input_Q')
    dssm_input_D = T.matrix('dssm_input_D')
    # ... and the desired output
#    mlp_target = T.col('mlp_target')
    # Learning rate and momentum hyperparameter values
    # Again, for non-toy problems these values can make a big difference
    # as to whether the network (quickly) converges on a good local minimum.
    learning_rate = 0.1
    momentum = 0.0
    # Create a function for computing the cost of the network given an input
    
    """
#    cost = mlp.squared_error(mlp_input, mlp_target)
    cost = dssm.output_train(dssm_index_Q, dssm_index_D, dssm_input_Q, dssm_input_D)
    cost_test = dssm.output_test(dssm_index_Q, dssm_index_D, dssm_input_Q, dssm_input_D)
        
    # Create a theano function for training the network
    train = theano.function([dssm_index_Q, dssm_index_D, dssm_input_Q, dssm_input_D], cost,
                            updates=gradient_updates_momentum(cost, dssm.params, learning_rate, momentum), mode=functionmode)
    # Create a theano function for computing the MLP's output given some input
    dssm_output = theano.function([dssm_index_Q, dssm_index_D, dssm_input_Q, dssm_input_D], cost_test, mode=functionmode)
    
    """
    ywcost = dssm.output_train_test(dssm_index_Q, dssm_index_D, dssm_input_Q, dssm_input_D)
    ywtest = theano.function([dssm_index_Q, dssm_index_D, dssm_input_Q, dssm_input_D], ywcost,
                             updates=gradient_updates_momentum(ywcost, dssm.params, learning_rate, momentum), mode=functionmode)
    
    # Keep track of the number of training iterations performed
    
    
    iteration = 1
    while iteration <= max_iteration:
        print "Iteration %d--------------" % (iteration)
        print "Each iteration contains %d minibatches" % (inputstream1.nTotalBatches)
        
        trainLoss = 0.0

        if inputstream1.BatchSize == inputstream1.minibatches[-1].SegSize:
            usefulbatches = inputstream1.nTotalBatches
        else:
            usefulbatches = inputstream1.nTotalBatches -1
        print "After removing the last incomplete batch, we need to process %d batches" % (usefulbatches)

        curr_minibatch1 = np.zeros((inputstream1.BatchSize, init_model_1.in_num_list[0]), dtype = numpy.float32)
        curr_minibatch2 = np.zeros((inputstream2.BatchSize, init_model_2.in_num_list[0]), dtype = numpy.float32)

        # we scan all minibatches, except the last one  
        for i in range(usefulbatches):
            inputstream1.setaminibatch(curr_minibatch1, i)
            inputstream2.setaminibatch(curr_minibatch2, i)
            
            current_output = ywtest(indexes[0], indexes[1], curr_minibatch1, curr_minibatch2)
            print "batch no %d, %f\n" % (i, current_output)
            trainLoss += current_output

        """        
        # For the last minibatch, train it only if it is a full batch
        i = inputstream1.nTotalBatches-1
        if inputstream1.BatchSize == inputstream1.minibatches[i].SegSize:
#            assert(inputstream1.BatchSize > inputstream1.minibatches[i].SegSize)
#            curr_minibatch1 = np.zeros((inputstream1.minibatches[i].SegSize, inputstream1.nMaxFeatureId+1), dtype = numpy.float32)
#            curr_minibatch2 = np.zeros((inputstream1.minibatches[i].SegSize, inputstream2.nMaxFeatureId+1), dtype = numpy.float32)
            inputstream1.setaminibatch(curr_minibatch1, i)
            inputstream2.setaminibatch(curr_minibatch2, i)
            
            current_output = ywtest(indexes[0], indexes[1], curr_minibatch1, curr_minibatch2)
            print "batch no %d, %f\n" % (i, current_output)
        """
        print "all batches in this iteraton is processed"
        print "trainLoss = %f" % (trainLoss)
                     
        # dump out current model separately
        tmpparams = []
        for W in dssm.params_Q:
            tmpparams.append(W.get_value())
        outfilename_1 = os.path.join(outputdir, "yw_dssm_Q_%d" % (iteration))
        save_simpledssmmodel(outfilename_1, SimpleDSSMModelFormat(init_model_1.mlayer_num, init_model_1.layer_info, init_model_1.mlink_num, init_model_1.in_num_list, init_model_1.out_num_list, tmpparams))
        

        tmpparams = []
        for W in dssm.params_D:
            tmpparams.append(W.get_value())
        outfilename_2 = os.path.join(outputdir, "yw_dssm_D_%d" % (iteration))
        save_simpledssmmodel(outfilename_2, SimpleDSSMModelFormat(init_model_2.mlayer_num, init_model_2.layer_info, init_model_2.mlink_num, init_model_2.in_num_list, init_model_2.out_num_list, tmpparams))
        
        print "Iteration %d-------------- is finished" % (iteration)
        
        iteration += 1

    print "-----The whole train process is finished-------\n"
    
def train_dssm_with_minibatch_predictiononly(bin_file_test_1, bin_file_test_2, dssm_file_1_simple, dssm_file_2_simple, outputfilename):
    
    # 0. open the outputfile
    outfile = open(outputfilename, 'w')
    
    # 1. Load in the input streams
    # Suppose the max seen feaid in the stream is 48930
    # then, inputstream1.nMaxFeatureId is 48931, which is one more
    # Ideally, the num_cols should be 48931
    # However, to make it conpatible with MS DSSM toolkit, we add it by one
    inputstream1 = InputStream(bin_file_test_1) # this will load in the whole file as origin. No modification at all
    inputstream2 = InputStream(bin_file_test_2)
    
    # 2. Load in the network structure and initial weights from DSSM
    init_model_1 = load_simpledssmmodel(dssm_file_1_simple)
    activations_1 = [T.tanh] * init_model_1.mlink_num
    
    init_model_2 = load_simpledssmmodel(dssm_file_2_simple)
    activations_2 = [T.tanh] * init_model_2.mlink_num

    # 3. Generate useful index structures
    # We assue that each minibatch is of the same size, i.e. mbsize
    # if the last batch has fewer samples, just ignore it
    # For prediction, we only need pairwise indexes
    mbsize = inputstream1.BatchSize
    ntrial = 1 # dumb parameter
    shift = 1 # dumb parameter
#    indexes = generate_index(mbsize, ntrial, shift) # for a normal minibatch, we should use this indexes
    indexes = [range(mbsize), range(mbsize)]
#    indexes_lastone = generate_index(inputstream1.minibatches[-1].SegSize, ntrial, shift) # this is used for the last batch

    # 4. Generate an instance of DSSM    
    dssm = DSSM(init_model_1.params, activations_1, init_model_2.params, activations_2, mbsize, ntrial, shift )

    # Create Theano variables for the MLP input
    dssm_index_Q = T.ivector('dssm_index_Q')
    dssm_index_D = T.ivector('dssm_index_D')
    dssm_input_Q = T.matrix('dssm_input_Q')
    dssm_input_D = T.matrix('dssm_input_D')
    # ... and the desired output
#    mlp_target = T.col('mlp_target')
    # Learning rate and momentum hyperparameter values
    # Again, for non-toy problems these values can make a big difference
    # as to whether the network (quickly) converges on a good local minimum.
#    learning_rate = 0.1
#    momentum = 0.0
    # Create a function for computing the cost of the network given an input
#    cost = mlp.squared_error(mlp_input, mlp_target)
#    cost = dssm.output_train(dssm_index_Q, dssm_index_D, dssm_input_Q, dssm_input_D)
    cost_test = dssm.output_test(dssm_index_Q, dssm_index_D, dssm_input_Q, dssm_input_D)
        
    # Create a theano function for training the network
#    train = theano.function([dssm_index_Q, dssm_index_D, dssm_input_Q, dssm_input_D], cost,
#                            updates=gradient_updates_momentum(cost, dssm.params, learning_rate, momentum), mode=functionmode)
    # Create a theano function for computing the MLP's output given some input
    dssm_output = theano.function([dssm_index_Q, dssm_index_D, dssm_input_Q, dssm_input_D], cost_test, mode=functionmode)
    
    
#    ywcost = dssm.output_train_test(dssm_index_Q, dssm_index_D, dssm_input_Q, dssm_input_D)
#    ywtest = theano.function([dssm_index_Q, dssm_index_D, dssm_input_Q, dssm_input_D], ywcost,
#                             updates=gradient_updates_momentum(ywcost, dssm.params, learning_rate, momentum), mode=functionmode)
    
    # Keep track of the number of training iterations performed
    
    
    iteration = 0
    while iteration < 1:
#        print "Iteration %d--------------" % (iteration)
        print "Each iteration contains totally %d minibatches" % (inputstream1.nTotalBatches)

        curr_minibatch1 = np.zeros((inputstream1.BatchSize, init_model_1.in_num_list[0]), dtype = numpy.float32)
        curr_minibatch2 = np.zeros((inputstream2.BatchSize, init_model_2.in_num_list[0]), dtype = numpy.float32)

        result = []
        
        # we scan all minibatches  
        for i in range(inputstream1.nTotalBatches):
            inputstream1.setaminibatch(curr_minibatch1, i)
            inputstream2.setaminibatch(curr_minibatch2, i)
            
#            current_output = ywtest(indexes[0], indexes[1], curr_minibatch1, curr_minibatch2)
            current_output = dssm_output(indexes[0], indexes[1], curr_minibatch1, curr_minibatch2)
#            current_output_list = current_output.tolist()
            print "batch no %d" % (i)
            tmplist = current_output[:, 0]
            result.extend(tmplist)
            

        """        
        # For the last minibatch, we might need to truncate it
        i = inputstream1.nTotalBatches-1
        inputstream1.setaminibatch(curr_minibatch1, i)
        inputstream2.setaminibatch(curr_minibatch2, i)
        current_output = dssm_output(indexes[0], indexes[1], curr_minibatch1, curr_minibatch2)
        print "batch no %d" % (i)
        if inputstream1.BatchSize == inputstream1.minibatches[i].SegSize:
#            result.append(current_output)
            tmplist = current_output[:, 0]
            result.extend(tmplist)
        else:
#            result.append(current_output[0:inputstream1.minibatches[i].SegSize, 0])
            tmplist = current_output[0:inputstream1.minibatches[i].SegSize, 0]
            result.extend(tmplist)
        """
        print "all batches in this iteraton is processed"
        
        for score in result:
            if math.isnan(score):
                break
            outfile.write(str(score))
            outfile.write("\n")
                         
                     
#        pickle.dump(result, outfile)
                 
        iteration += 1

    outfile.close()

def test_dssm_with_minibatch_prediction(bin_file_train_1, bin_file_train_2, bin_file_test_1, bin_file_test_2, dssm_file_1, dssm_file_2, outputdir, ntrial, shift):
    
    f = open("/home/yw/Downloads/test.2.bin", "rb")
    g = open("/home/yw/Downloads/test.2.bin", "rb")
    
    # 1. get the last five numbers
    # nMaxFeatureId, nLine, nMaxSegmentSize, nMaxFeatureNum, BatchSize
    f.seek(-20, 2)
    c = np.fromfile(f, dtype=np.uint32)
    inputstream1 = InputStream(c)
    
    # 2. load in all minibatches into 
    inputstream1.loadinallminibatches(f)
#    inputstream1.display()

    # 3. Get dimension of a minibatch
    curr_minibatch1 = np.zeros((inputstream1.BatchSize, inputstream1.nMaxFeatureId), dtype = numpy.float32)
#    print curr_minibatch

    g.seek(-20, 2)
    d = np.fromfile(g, dtype=np.uint32)
    inputstream2 = InputStream(d)
    inputstream2.loadinallminibatches(g)
    curr_minibatch2 = np.zeros((inputstream2.BatchSize, inputstream2.nMaxFeatureId), dtype = numpy.float32)

    # Training data - two randomly-generated Gaussian-distributed clouds of points in 2d space
    np.random.seed(0)
    
    
    layer_sizes = [1765, 2]#, X.shape[1]*2]#, X.shape[1]*2, X.shape[1]*2]
    W_init = []
    
    z = open('/home/yw/Downloads/dssm_0', 'rb')
    e = pickle.load(z)
    b1 = pickle.load(z)
    b2 = pickle.load(z)
    W_init = [b1, b2]
    z.close()

#    b_init = []
    activations = []
    for n_input, n_output in zip(layer_sizes[:-1], layer_sizes[1:]):
        activations.append(T.tanh)
    # Create an instance of the MLP class
    mbsize = inputstream1.BatchSize
    neg = 1
    shift = 1
    indexes = generate_index(mbsize, neg, shift)
#    print indexes
#    print indexes[0].dtype
    
    dssm = DSSM(W_init, mbsize, neg, shift, activations, 1)
    
    print "W_init is as follows:"
    print W_init
    
#    print "b_init is as follows:"
#    print b_init
    
    # Create Theano variables for the MLP input
    dssm_index_Q = T.ivector('dssm_index_Q')
    dssm_index_D = T.ivector('dssm_index_D')
    dssm_input_Q = T.matrix('dssm_input_Q')
    dssm_input_D = T.matrix('dssm_input_D')
    # ... and the desired output
#    mlp_target = T.col('mlp_target')
    # Learning rate and momentum hyperparameter values
    # Again, for non-toy problems these values can make a big difference
    # as to whether the network (quickly) converges on a good local minimum.
    learning_rate = 0.01
    momentum = 0.9
    # Create a function for computing the cost of the network given an input
#    cost = mlp.squared_error(mlp_input, mlp_target)
    cost = dssm.output_train(dssm_index_Q, dssm_index_D, dssm_input_Q, dssm_input_D)
    # Create a theano function for training the network
    train = theano.function([dssm_index_Q, dssm_index_D, dssm_input_Q, dssm_input_D], cost,
                            updates=gradient_updates_momentum(cost, dssm.params, learning_rate, momentum), mode=functionmode)

    
    
    cost_test = dssm.output_test(dssm_index_Q, dssm_index_D, dssm_input_Q, dssm_input_D)
    dssm_output = theano.function([dssm_index_Q, dssm_index_D, dssm_input_Q, dssm_input_D], cost_test, mode=functionmode)
    
    
    ywcost = dssm.output_train_test(dssm_index_Q, dssm_index_D, dssm_input_Q, dssm_input_D)
    ywtest = theano.function([dssm_index_Q, dssm_index_D, dssm_input_Q, dssm_input_D], ywcost,
                             updates=gradient_updates_momentum(ywcost, dssm.params, learning_rate, momentum), mode=functionmode)
    
    # Keep track of the number of training iterations performed
    for i in range(inputstream1.nTotalBatches):
        inputstream1.setaminibatch(curr_minibatch1, i)
        inputstream2.setaminibatch(curr_minibatch2, i)
        
        current_output = dssm_output(indexes[2], indexes[3], curr_minibatch1, curr_minibatch2)
        print current_output
        
    
    
def test_file1():
    f = open('/home/yw/Downloads/dssm_1', 'rb')
    e = pickle.load(f)
    b1 = pickle.load(f)
    b2 = pickle.load(f)

    print e, b1, b2
    
    f.close()
    
def test_file():
    d = dict(name='Bob', age=20, score=88)
    a1 = np.zeros((3,5), dtype=np.int8)
    a2 = np.zeros((4,4), dtype=np.float32)
    a3 = np.ones((2,2), dtype=np.float32)
    
#    print d, a1, a2, a3
    f = open('/home/yw/Downloads/dump.txt', 'wb')
    pickle.dump(d, f)
    pickle.dump(a1, f)
    pickle.dump(a2, f)
    pickle.dump(a3, f)
    f.close()
    
    f = open('/home/yw/Downloads/dump.txt', 'rb')
    e = pickle.load(f)
    b1 = pickle.load(f)
    b2 = pickle.load(f)
    b3 = pickle.load(f)
    
    f.close()
    print e, b1, b2, b3

    

def test_load_bin_file():
    f = open("/home/yw/Downloads/test.2.bin", "rb")
    g = open("/home/yw/Downloads/test.2.bin", "rb")
    
    # 1. get the last five numbers
    # nMaxFeatureId, nLine, nMaxSegmentSize, nMaxFeatureNum, BatchSize
    f.seek(-20, 2)
    c = np.fromfile(f, dtype=np.uint32)
    inputstream1 = InputStream(c)
    
    # 2. load in all minibatches into 
    inputstream1.loadinallminibatches(f)
#    inputstream1.display()

    # 3. Get dimension of a minibatch
    curr_minibatch1 = np.zeros((inputstream1.BatchSize, inputstream1.nMaxFeatureId), dtype = numpy.float32)
#    print curr_minibatch

    g.seek(-20, 2)
    d = np.fromfile(g, dtype=np.uint32)
    inputstream2 = InputStream(d)
    inputstream2.loadinallminibatches(g)
    curr_minibatch2 = np.zeros((inputstream2.BatchSize, inputstream2.nMaxFeatureId), dtype = numpy.float32)


    
#    inputstream1.setaminibatch(curr_minibatch, 0)

#    print curr_minibatch
    for i in range(inputstream1.nTotalBatches):
        inputstream1.setaminibatch(curr_minibatch1, i)
        inputstream2.setaminibatch(curr_minibatch2, i)
        print 'i = %d' % (i)
        print curr_minibatch1
        print curr_minibatch2
#    def setaminibatch(self, curr_minibatch, i):

    
    
# input: a file name
# output: an ndarray of weights. For default DSSM, it is just a list of matrices
def test_load_dssmmodel(filename):
#    f = open("/home/yw/Downloads/dssm/WebSearch/config_WebSearch_FullyConnect.txt.train/DSSM_DOC_ITER0", "rb")
    f = open(filename, "rb")
    
    mlayer_num = np.fromfile(f, dtype=np.int32, count = 1)
    layer_info = np.fromfile(f, dtype=np.int32, count = mlayer_num)

    mlink_num = np.fromfile(f, dtype=np.int32, count = 1)
    in_num_list = []
    out_num_list = []
    for i in range(mlink_num):
        in_num, out_num = np.fromfile(f, dtype=np.int32, count = 2)
        initbias, initweight = np.fromfile(f, dtype=np.float32, count = 2)
        unusedlist = np.fromfile(f, dtype=np.int32, count = 3)
        in_num_list.append(in_num)
        out_num_list.append(out_num)
    
    params = []
    for i in range(mlink_num):
        weight_len =  np.fromfile(f, dtype=np.int32, count = 1)
        weights = np.fromfile(f, dtype=np.float32, count = weight_len)
        weights_matrix = np.reshape(weights, (in_num_list[i], out_num_list[i]))
        params.append(weights_matrix)
        
        bias_len =  np.fromfile(f, dtype=np.int32, count = 1)
        biases = np.fromfile(f, dtype=np.float32, count = bias_len)
    f.close()   

    return [layer_info, params]

# input: a file name of the Microsoft DSSM model file
# output: a simpler file using Pickle, only have layer_info (an array) and a list of Ws
def convert_microsoftdssmmodel(infilename, outfilename):
#    f = open("/home/yw/Downloads/dssm/WebSearch/config_WebSearch_FullyConnect.txt.train/DSSM_DOC_ITER0", "rb")
    f = open(infilename, "rb")
    
    mlayer_num = np.fromfile(f, dtype=np.int32, count = 1)
    layer_info = np.fromfile(f, dtype=np.int32, count = mlayer_num)

    mlink_num = np.fromfile(f, dtype=np.int32, count = 1)
    in_num_list = []
    out_num_list = []
    for i in range(mlink_num):
        in_num, out_num = np.fromfile(f, dtype=np.int32, count = 2)
        initbias, initweight = np.fromfile(f, dtype=np.float32, count = 2)
        unusedlist = np.fromfile(f, dtype=np.int32, count = 3)
        in_num_list.append(in_num)
        out_num_list.append(out_num)
    
    params = []
    for i in range(mlink_num):
        weight_len =  np.fromfile(f, dtype=np.int32, count = 1)
        weights = np.fromfile(f, dtype=np.float32, count = weight_len)
        weights_matrix = np.reshape(weights, (in_num_list[i], out_num_list[i]), order='C') # weights is row based
        params.append(weights_matrix)
        
        bias_len =  np.fromfile(f, dtype=np.int32, count = 1)
        biases = np.fromfile(f, dtype=np.float32, count = bias_len)
    f.close()   

#    return [layer_info, params]
    model = SimpleDSSMModelFormat(mlayer_num[0], layer_info, mlink_num[0], in_num_list, out_num_list, params)

    h = open(outfilename, 'wb')
    pickle.dump(model, h)
#    pickle.dump(mlayer_num, h)
#    pickle.dump(layer_info, h)
#    pickle.dump(mlink_num, h)
#    pickle.dump(in_num_list, h)
#    pickle.dump(out_num_list, h)
#    pickle.dump(params, h)
    h.close()

def load_simpledssmmodel(infilename):
    h = open(infilename, 'rb')
    model = pickle.load(h)
    h.close()
    return model

def save_simpledssmmodel(filename, model):
    h = open(filename, 'wb')
    pickle.dump(model, h)
    h.close()


if __name__ == '__main__':
#    test_dssm()
#    test_file1()
#    test_dssm_with_minibatch_prediction()
#    test_load_bin_file()    
    basedir_data = "/home/yw/Documents/sigir2015/Dataset/toy03"
    basedir_initmodel = "/home/yw/Documents/sigir2015/Experiments/toy03/WebSearch/config_WebSearch_FullyConnect.txt.train"
    bin_file_train_1 = "%s/train.1.src.seq.bin" % (basedir_data)
    bin_file_train_2 = "%s/train.1.tgt.seq.bin" % (basedir_data)
    bin_file_test_1 = "%s/valid.1.src.seq.bin" % (basedir_data)
    bin_file_test_2 = "%s/valid.1.tgt.seq.bin" % (basedir_data)
    dssm_file_1 = "%s/DSSM_QUERY_ITER0" % (basedir_initmodel)
    dssm_file_2 = "%s/DSSM_DOC_ITER0" % (basedir_initmodel)
    outputdir = "/home/yw/Documents/sigir2015/Experiments/toy03/WebSearch/yw.train"
    ntrial = 1
    shift = 1
    max_iteration = 5

    dssm_file_1_simple = "%s_simple" % (dssm_file_1)
    dssm_file_2_simple = "%s_simple" % (dssm_file_2)

    """
    # the following loop is convert the original dssm model to simple format
    for i in range(-1):    
        dssm_file_1 = "%s/DSSM_QUERY_ITER%d" % (basedir_initmodel, i)
        dssm_file_1_simple = "%s_simple" % (dssm_file_1)
        convert_microsoftdssmmodel(dssm_file_1, dssm_file_1_simple)
        dssm_file_2 = "%s/DSSM_DOC_ITER%d" % (basedir_initmodel, i)
        dssm_file_2_simple = "%s_simple" % (dssm_file_2)
        convert_microsoftdssmmodel(dssm_file_2, dssm_file_2_simple)
    
#        dssm_file_1_simple = "%s_simple" % (dssm_file_1)
#        convert_microsoftdssmmodel(dssm_file_1, dssm_file_1_simple)
#        dssm_file_2_simple = "%s_simple" % (dssm_file_2)
#        convert_microsoftdssmmodel(dssm_file_2, dssm_file_2_simple)
    

    # the following loop is to conduct prediction using dssm model (simple version)
    for i in range(-1):#    i = 0
        dssm_file_1 = "%s/DSSM_QUERY_ITER%d" % (basedir_initmodel, i)
        dssm_file_1_simple = "%s_simple" % (dssm_file_1)
        dssm_file_2 = "%s/DSSM_DOC_ITER%d" % (basedir_initmodel, i)
        dssm_file_2_simple = "%s_simple" % (dssm_file_2)
        outputfilename = "%s_prediction" % (dssm_file_1_simple)
    
        train_dssm_with_minibatch_predictiononly(bin_file_train_1, bin_file_train_2, dssm_file_1_simple, dssm_file_2_simple, outputfilename)
    """
    
    
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)
        
    train_dssm_with_minibatch(bin_file_train_1, bin_file_train_2, dssm_file_1_simple, dssm_file_2_simple, outputdir, ntrial, shift, max_iteration)    

    sys.exit(0)

    for i in range(1, 6):
        dssm_file_1_predict = "%s/yw_dssm_Q_%d" % (outputdir, i)
        dssm_file_2_predict = "%s/yw_dssm_D_%d" % (outputdir, i)
#        outputfilename = "%s/yw_dssm_Q_%d_prediction" % (outputdir, i)
        outputfilename = "%s_prediction" % (dssm_file_1_predict)
    
        train_dssm_with_minibatch_predictiononly(bin_file_train_1, bin_file_train_2, dssm_file_1_predict, dssm_file_2_predict, outputfilename)


    print '----------------finished--------------------'
    bin_file_train_1 = "/home/yw/Downloads/dssm/train_4M_test_900K/WebSearch/train.QT.pair.1.src.seq.fea.bin"
    bin_file_train_2 = "/home/yw/Downloads/dssm/train_4M_test_900K/WebSearch/train.QT.pair.1.tgt.seq.fea.bin"
    bin_file_test_1 = "/home/yw/Downloads/dssm/train_4M_test_900K/WebSearch/test.QT.pair.1.src.seq.fea.bin"
    bin_file_test_2 = "/home/yw/Downloads/dssm/train_4M_test_900K/WebSearch/test.QT.pair.1.tgt.seq.fea.bin"
    dssm_file_1 = "/home/yw/Downloads/dssm/train_4M_test_900K/WebSearch/DSSM_QUERY_ITER0"
    dssm_file_2 = "/home/yw/Downloads/dssm/train_4M_test_900K/WebSearch/DSSM_DOC_ITER0"
    outputdir = "/home/yw/Downloads/dssm/train_4M_test_900K/WebSearch/yw"
    ntrial = 50
    shift = 1


    basedir_data = "/home/yw/Documents/sigir2015/Dataset/train_10K_test_1K/WebSearch"
    basedir_initmodel = "/home/yw/Documents/sigir2015/Experiments/toy02/WebSearch/config_WebSearch_FullyConnect.txt.train"
    bin_file_train_1 = "%s/train.1.src.seq.bin" % (basedir_data)
    bin_file_train_2 = "%s/train.1.tgt.seq.bin" % (basedir_data)
    bin_file_test_1 = "%s/valid.1.src.seq.bin" % (basedir_data)
    bin_file_test_2 = "%s/valid.1.tgt.seq.bin" % (basedir_data)
    dssm_file_1 = "%s/DSSM_QUERY_ITER0" % (basedir_initmodel)
    dssm_file_2 = "%s/DSSM_DOC_ITER0" % (basedir_initmodel)
    outputdir = "/home/yw/Documents/sigir2015/Experiments/toy02/WebSearch/yw.train"
    ntrial = 1
    shift = 1
    max_iteration = 5
