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

from theano import tensor

try:
    import cPickle as pickle
except ImportError:
    import pickle
    
functionmode = 'DebugMode'
functionmode = 'FAST_RUN'

class ParameterSetting(object):
    def __init__(self, configfilename):
        # parameters should be consistent with the dssm config file
        self.shift = 1
        
        f = open(configfilename, "r")
        for line in f:
            fields = line[:-2].split('\t')
            
            if fields[0] == "QFILE":
                self.bin_file_train_1 = fields[1]
                continue
            elif fields[0] == "DFILE":
                self.bin_file_train_2 = fields[1]
                continue
            elif fields[0] == "MODELPATH":
#                pos = fields[1].rfind('\\')
                self.outputdir = fields[1]
                continue
            elif fields[0] == "SEEDMODEL1":
                # we need to convert dssm model from original format
                self.dssm_file_1_simple = fields[1]
#                self.dssm_file_1_simple = "%s_simple" % (dssm_file_1)
#                if not os.path.exists(self.dssm_file_1_simple):
#                    convert_microsoftdssmmodel(dssm_file_1, self.dssm_file_1_simple)
                continue
            elif fields[0] == "SEEDMODEL2":
                # we need to convert dssm model from original format
                self.dssm_file_2_simple = fields[1]
#                self.dssm_file_2_simple = "%s_simple" % (dssm_file_2)
#                if not os.path.exists(self.dssm_file_2_simple):
#                    convert_microsoftdssmmodel(dssm_file_2, self.dssm_file_2_simple)
                continue
            elif fields[0] == "VALIDATEQFILE":
                self.bin_file_test_1 = fields[1]
                continue
            elif fields[0] == "VALIDATEDFILE":
                self.bin_file_test_2 = fields[1]
                continue
            elif fields[0] == "VALIDATEPAIR":
                self.labelfile = fields[1]
                continue
            elif fields[0] == "NTRIAL":
                self.ntrial = int(fields[1])
                continue
            elif fields[0] == "MAX_ITER":
                self.max_iteration = int(fields[1])
                continue
        f.close()
         

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

    def output(self, index_Q, index_D, Q, D):   
        # components is a vector         
        components, updates = theano.scan(self.ComputeCosineBetweenTwoVectors,
                                  outputs_info=None,
                                  sequences=[index_Q, index_D],
                                  non_sequences=[Q,D])
        
        
        # get the final output, which is a list of Tvariables
        return components

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
 

    def output_train(self, index_Q, index_D, Q, D):
        '''
        Compute the DSSM's output given an input
        
        :parameters:
            - index_Q, index_D : each is a list of integers, i.e. two tensor vectors
                two indexes for corresponding vectors

            - Q,D : theano.tensor.var.TensorVariable, should be two matrices
                Theano symbolic variable for layer input

        :returns:
            - output : theano.tensor.var.TensorVariable, should be a tensor scalar, which serves as the train loss
                A scalar value
        '''
        # Recursively compute output
        for layer in self.layers_Q:
            Q = layer.output(Q)
        for layer in self.layers_D:
            D = layer.output(D)
#        return Q, D
        
#        cosine_matrix = self.layer_cosine.output(index_Q, index_D, Q, D) * 10 # scaling by 10 is suggested by Jianfeng Gao
        cosine_matrix = self.layer_cosine.output_noloop_1(index_Q, index_D, Q, D) * 10 # scaling by 10 is suggested by Jianfeng Gao
    
        cosine_matrix_reshape = T.reshape(cosine_matrix, (self.layer_cosine.n_mbsize, self.layer_cosine.n_neg+1))
        
        # for this matrix, each line is a prob distribution right now.
        cosine_matrix_reshape_softmax = T.nnet.softmax(cosine_matrix_reshape)
#        return cosine_matrix_reshape_softmax
        
        # get the first column, i.e. the column of positive pairs (Q,D)
        column1 = cosine_matrix_reshape_softmax[:,0]
        
        column1_neglog = -T.log(column1)

        return column1_neglog.sum()
#        return column1_neglog
        
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
            - output : theano.tensor.var.TensorVariable, should be a tensor column vector
        '''
        # Recursively compute output
        for layer in self.layers_Q:
            Q = layer.output(Q)
        for layer in self.layers_D:
            D = layer.output(D)
        
        cosine_matrix = self.layer_cosine.output_noloop_1(index_Q, index_D, Q, D)
        cosine_matrix_reshape = T.reshape(cosine_matrix, (self.layer_cosine.n_mbsize, 1))
        
        return cosine_matrix_reshape


def gradient_updates_momentum(cost, params, learning_rate, momentum, mbsize):
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
        updates.append((param, param - (learning_rate*theano.grad(cost, param) / float(mbsize))))
    return updates

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
    """
    ywcost = dssm.output_train(dssm_index_Q, dssm_index_D, dssm_input_Q, dssm_input_D)
    ywtest = theano.function([dssm_index_Q, dssm_index_D, dssm_input_Q, dssm_input_D], ywcost,
                             updates=gradient_updates_momentum(ywcost, dssm.params, learning_rate, momentum), mode=functionmode)
    """
    ywcost = dssm.output_train(dssm_index_Q, dssm_index_D, dssm_input_Q, dssm_input_D)
#    ywcost_scalar = ywcost.sum()
    ywtest = theano.function([dssm_index_Q, dssm_index_D, dssm_input_Q, dssm_input_D], ywcost,
                             updates=gradient_updates_momentum(ywcost, dssm.params, learning_rate, momentum, mbsize), mode=functionmode)
    # Keep track of the number of training iterations performed
    grad_ywcost = theano.grad(ywcost, dssm.params)
    grad_ywtest = theano.function([dssm_index_Q, dssm_index_D, dssm_input_Q, dssm_input_D], grad_ywcost, mode=functionmode)
 
    
    
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
            print "Processing batch no %d\n" % (i)
            
            inputstream1.setaminibatch(curr_minibatch1, i)
            inputstream2.setaminibatch(curr_minibatch2, i)

#            grad_current_output =  grad_ywtest(indexes[0], indexes[1], curr_minibatch1, curr_minibatch2)           
            current_output = ywtest(indexes[0], indexes[1], curr_minibatch1, curr_minibatch2)
#            print "batch no %d, %f\n" % (i, current_output)
#           print current_output
            trainLoss += current_output

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
    
def train_dssm_with_minibatch_gradient_check(bin_file_train_1, bin_file_train_2, dssm_file_1_simple, dssm_file_2_simple, outputdir, ntrial, shift, max_iteration):
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
    """
    ywcost = dssm.output_train(dssm_index_Q, dssm_index_D, dssm_input_Q, dssm_input_D)
    ywtest = theano.function([dssm_index_Q, dssm_index_D, dssm_input_Q, dssm_input_D], ywcost,
                             updates=gradient_updates_momentum(ywcost, dssm.params, learning_rate, momentum), mode=functionmode)
    """
    ywcost = dssm.output_train(dssm_index_Q, dssm_index_D, dssm_input_Q, dssm_input_D)
    grad_ywcost = theano.grad(ywcost, dssm.params)
    
    ywtest = theano.function([dssm_index_Q, dssm_index_D, dssm_input_Q, dssm_input_D], ywcost, mode=functionmode)
    grad_ywtest = theano.function([dssm_index_Q, dssm_index_D, dssm_input_Q, dssm_input_D], grad_ywcost, mode=functionmode)
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
            
            current_output = grad_ywtest(indexes[0], indexes[1], curr_minibatch1, curr_minibatch2)
#            current_output /= (-2)
#            print "batch no %d, %f\n" % (i, current_output)
            print "batch no %d\n" % (i)
            """
            for tmp_output in current_output:
                tmp_list = []
                for m in range(5):
                    for n in range(2):
                        tmp_list.append(tmp_output[m,n])
                print tmp_list
            """
#            print current_output
#            trainLoss += current_output
            for k in range(len(dssm.params)):
                base_w = dssm.params[k].get_value() # get current weights
                base_w_new = base_w - 0.1*current_output[k]
                dssm.params[k].set_value(base_w_new)
                print base_w
                print current_output[k]
                print base_w_new

            """
            # Next, we need to manually compute gradient
            for k in range(len(dssm.params)):
                base_w = dssm.params[k].get_value()
                grad_base_w = base_w.copy()
                eps = 0.0001
                
                for x in range(5):
                    for y in range(2):
                        base_w_plus = base_w.copy()
                        base_w_plus[x,y] += eps
                        base_w_neg = base_w.copy()
                        base_w_neg[x,y] -= eps

                        dssm.params[k].set_value(base_w_plus)
                        current_output_plus = ywtest(indexes[0], indexes[1], curr_minibatch1, curr_minibatch2)
                        dssm.params[k].set_value(base_w_neg)
                        current_output_neg = ywtest(indexes[0], indexes[1], curr_minibatch1, curr_minibatch2)
                        final_grad = (current_output_plus-current_output_neg)/(2*eps)
                        grad_base_w[x,y] = final_grad
#                print grad_base_w.tolist()
                # first, dump out the theano.grad result
                tmp_list = []
                for m in range(5):
                    for n in range(2):
                        tmp_list.append(current_output[k][m,n])
                print tmp_list

                
                
                tmp_list = []
                for m in range(5):
                    for n in range(2):
                        tmp_list.append(grad_base_w[m,n])
                print tmp_list
                dssm.params[k].set_value(base_w)
            """
                        
            """                        
            (x,y) = (0,0)
            base_WQ = dssm.params_Q[0].get_value()
            base_WQ_Plus = base_WQ.copy()
            base_WQ_Plus[x,y] += eps
            base_WQ_Neg = base_WQ.copy()
            base_WQ_Neg[x,y] -= eps
            """
            
            # compute plus

            # set back the original weight            
#            dssm.params_Q[0].set_value(base_WQ)
            
            
            
            

        print "all batches in this iteraton is processed"
        print "trainLoss = %f" % (trainLoss)
                     
        print "Iteration %d-------------- is finished" % (iteration)
        
        iteration += 1

    print "-----The whole train process is finished-------\n"
    


def train_dssm_with_minibatch_predictiononly(bin_file_test_1, bin_file_test_2, dssm_file_1_simple, dssm_file_2_simple, labelfilename, outputfilename):
    
    # 0. open the outputfile
    outfile = open(outputfilename, 'w')
    labelfile = open(labelfilename, 'r')
    
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
    
    
#        print "Iteration %d--------------" % (iteration)
    print "This prediction contains totally %d minibatches" % (inputstream1.nTotalBatches)

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
        

    print "all batches in this iteraton is processed"
    
    line_labelfile = labelfile.readline()
    line_output = line_labelfile[:-1] + "\tDSSM\n"
    outfile.write(line_output)
    
    for score in result:
        if math.isnan(score):
            break
        
        line_labelfile = labelfile.readline()
        line_output = "%s\t%f\n" % (line_labelfile[:-1], score)
#        outfile.write(str(score))
        outfile.write(line_output)
                     

    outfile.close()
    labelfile.close()
    
    
    
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


def func_ComputeCosineMatrix(Q, inds_Q, D, inds_D):
    """
    Input: Q is T.fmatrix(), shape=(bs, eb)
    Input: indx_Q is T.ivector(), shape = (4,)
    Input: D is T.fmatrix(), shape = (bs, eb)
    Input: indx_D is T.ivector(), shape = (4,)
    Output: QPN, DPN
    """
    
#    inds = [0,1,0,1]
#    return D[inds_D]

    Q_view = Q[inds_Q]
    D_view = D[inds_D]
    
    
    dotQD = Q_view * D_view #  Q[inds_Q]*D[inds_D]
    dotQD = tensor.sum(dotQD, axis = dotQD.ndim-1)#, keepdims=True)
    
    dotQQ = Q_view * Q_view
    dotQQ = tensor.sum(dotQQ, axis = dotQQ.ndim-1)
    
    dotDD = D_view * D_view
    dotDD = tensor.sum(dotDD, axis = dotDD.ndim-1)
    
    dotQQDD = dotQQ*dotDD
    
    dotQQDD_sqrt = tensor.sqrt(dotQQDD)
    
#    q_norm = 
    return dotQD, dotQQ, dotDD, dotQQDD, dotQQDD_sqrt, dotQD/dotQQDD_sqrt
    
#    return Q[inds_Q]*D[inds_D]
#tensor.sum(x, axis=x.ndim-1, keepdims=True)


def test_broadcasts3():
    

    Q = tensor.fmatrix()
    D = tensor.fmatrix()
    inds_Q = tensor.ivector() 
    inds_D = tensor.ivector() 
    DPN = func_ComputeCosineMatrix(Q, inds_Q, D, inds_D)
    
    func_ComputeCosineMatrix_real = theano.function([Q, inds_Q, D, inds_D], DPN)
 
    Q_value = numpy.array([[1,1], [2,3]]).astype("float32")
    D_value = numpy.array([[4,5], [4,6]]).astype("float32")
    
    inds_Q_value = numpy.array([1, 1, 0, 0]).astype("int32")
    inds_D_value = numpy.array([0, 1, 0, 1]).astype("int32")
    
    result = func_ComputeCosineMatrix_real(Q_value, inds_Q_value, D_value, inds_D_value)
    print result
    
#    W = theano.shared(value = numpy.array([[1, 2, 3], [3, 4, 5]]).astype("float32"))
    
    
    

    
     

def test_broadcasts2():
    W = theano.shared(value = numpy.array([[1, 2, 3], [3, 4, 5]]).astype("float32"))
    
    num_negs = 1
    batch_size = 1
    embeddings = 3
    Q = tensor.imatrix()
    D = tensor.imatrix()
    
    Q_e = W[Q].sum(1) # batch_size x embeddings, [[6 , 8, 10]]
    D_e = W[D].sum(1) # (num_negs + 1) * batch_size x embeddings, [[6, 8,    10], [4, 6, 8]]
    
    # Multiple layers here .....
    
    D_e = D_e.reshape((num_negs + 1, batch_size, embeddings))
    
    def softmax(x):
        x = tensor.exp(x - tensor.max(x, axis=x.ndim-1, keepdims=True))
        return x / tensor.sum(x, axis=x.ndim-1, keepdims=True)
    
    # compute dots
    dots = (D_e * Q_e).sum(2).dimshuffle(1,0) # [6*6 + 8*8 + 10*10 = 200,     6*4 + 8*6 + 10*8 = 152]
    loss = tensor.log(softmax(dots)[:, 0]).sum() # Sum always first     column, i.e. first example in D amongst the negatives is always the     positive one
    index = theano.function([Q, D], [Q_e, D_e, dots, loss])
    
    func_WQ = theano.function([Q], Q_e)
    func_WD = theano.function([D], D_e)
    func_dots = theano.function([Q, D], softmax(dots)) #tensor.exp(dots - tensor.max(dots, axis=dots.ndim-1, keepdims=True)))
    
    Q_value = [[1,1]]
    D_value = [[1,1],[0,1]]
    
#    print index([[1,1]], [[1,1],[0,1]])
#    print index(Q_value, D_value)
    result = func_dots(Q_value, D_value)
    print result
    print type(func_WQ(Q_value))
#    print func_WD(D_value)
#    print func_dots(Q_value, D_value)
    


def test_broadcasts1():
    Q = T.imatrix()
    D = T.itensor3()
    QD = Q * D
    func_dot = theano.function([Q, D], QD)
    
    Q_value = [[1,1], [2,2]]
    D_value = [ [[1,2,3], [4,5,6]],   [[1,2,3], [4,5,6]] ]
    print Q_value
    print D_value
    
    QD_value = func_dot(Q_value, D_value)
    print QD_value
    
    
    
    
    
    
    
def test_broadcasts():
    A = T.imatrix()
    A_S = A.dimshuffle(0, 'x',1)
    func_shuffle = theano.function([A], A_S)
    A_value = [[1,2], [3,4]]
    AS_value = func_shuffle(A_value)
    
    print A_value
    print AS_value
    print AS_value.shape
    
    
    B = T.itensor3()
    AB = A_S + B
    func_add = theano.function([A_S, B], AB)
    
    B_value = [ A_value,
                A_value]
    
    AB_value = func_add(AS_value, B_value)
    print AB_value.shape
    
    AA = A[[0,0,0,0]]
    func_embed = theano.function([A], AA)
    AA_value = func_embed(A_value)
    print AA_value
    

    """    
    
    C = T.matrix()
    D = B+C
    simpleadd = function([B, C], D)
    
    
    C_value = [ A_value,
                A_value]
    
    print C_value
    """
    
    


if __name__ == '__main__':
#    test_dssm()
#    test_file1()
#    test_dssm_with_minibatch_prediction()
#    test_load_bin_file()    
#    basedir_data = "/home/yw/Documents/sigir2015/Dataset/toy03"
#    basedir_initmodel = "/home/yw/Documents/sigir2015/Experiments/toy03/WebSearch/config_WebSearch_FullyConnect.txt.train"

#    test_broadcasts3()
#    sys.exit(0)

    
    if len(sys.argv) >=3:
#        print 'Argument List', str(sys.argv)
        if sys.argv[1] == "-convertmicrosoftdssmmodel":
            assert(len(sys.argv) == 4)
            print "We need to convert a dssm model file from Microsot Format to Simple Format"
            convert_microsoftdssmmodel(sys.argv[2], sys.argv[3])
        elif sys.argv[1] == "-train":
            assert(len(sys.argv) == 3)
            print "We need to train a dssm model fram the beginning"
            if not os.path.exists(sys.argv[2]):
                print sys.argv[2] + " doesn't exist!"
            ps = ParameterSetting(sys.argv[2])

            if not os.path.exists(ps.outputdir):
                os.makedirs(ps.outputdir)
        
            train_dssm_with_minibatch(ps.bin_file_train_1, ps.bin_file_train_2, ps.dssm_file_1_simple, ps.dssm_file_2_simple, ps.outputdir, ps.ntrial, ps.shift, ps.max_iteration)    
        #    train_dssm_with_minibatch_gradient_check(bin_file_train_1, bin_file_train_2, dssm_file_1_simple, dssm_file_2_simple, outputdir, ntrial, shift, max_iteration)(bin_file_train_1, bin_file_train_2, dssm_file_1_simple, dssm_file_2_simple, outputdir, ntrial, shift, max_iteration)    
        
        #    sys.exit(0)
        
            for i in range(ps.max_iteration+1):
                dssm_file_1_predict = "%s/yw_dssm_Q_%d" % (ps.outputdir, i)
                dssm_file_2_predict = "%s/yw_dssm_D_%d" % (ps.outputdir, i)
        #        outputfilename = "%s/yw_dssm_Q_%d_prediction" % (outputdir, i)
                outputfilename = "%s_prediction" % (dssm_file_1_predict)
            
                train_dssm_with_minibatch_predictiononly(ps.bin_file_test_1, ps.bin_file_test_2, dssm_file_1_predict, dssm_file_2_predict, ps.labelfile, outputfilename)
        
            
    
    else:
        print 'Error\n'
    sys.exit(0)


    print '----------------finished--------------------'


    """    
    basedir_data = "./Dataset/train_10K_test_1K/WebSearch"
#    basedir_data = "./Dataset/toy03"
    basedir_initmodel = "./Experiments/toy05/WebSearch/config_WebSearch_FullyConnect.txt.train"
    bin_file_train_1 = "%s/train.1.src.seq.fea.bin" % (basedir_data)
    bin_file_train_2 = "%s/train.1.tgt.seq.fea.bin" % (basedir_data)
    bin_file_test_1 = "%s/valid.1.src.seq.fea.bin" % (basedir_data)
    bin_file_test_2 = "%s/valid.1.tgt.seq.fea.bin" % (basedir_data)
    dssm_file_1 = "%s/DSSM_QUERY_ITER0" % (basedir_initmodel)
    dssm_file_2 = "%s/DSSM_DOC_ITER0" % (basedir_initmodel)
    outputdir = "./Experiments/toy05/WebSearch/yw.train2"
    labelfile = "./Dataset/train_10K_test_1K/WebSearch_rawdata/valid.label.1000.txt.unix"
    
    # parameters should be consistent with the dssm config file
    ntrial = 50
    shift = 1
    max_iteration = 5

    dssm_file_1_simple = "%s_simple" % (dssm_file_1)
    dssm_file_2_simple = "%s_simple" % (dssm_file_2)

    # the following loop is convert the original dssm model to simple format
    for i in range(1):    
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
