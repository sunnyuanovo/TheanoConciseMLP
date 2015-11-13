"""
This network structure is the same as Microsoft DSSM.
Both Q and D go through the same network, such as 49K-128-128
A final cosine layer is applied to the embedding results.
"""


import sys
sys.path.append('/u/yuanwei/workspace/TheanoConciseMLP/Utilities')  
sys.path.append('/home/yw/workspace/test/TheanoConciseMLP/Utilities')  
import basic_utilities
import test_utilities

import NetworkComponents as nc

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


class SimpleDSSMModel_0_Format(object):
    def __init__(self, params):
        """
        params == a list of weights, like  [W1, W2, W3]
        """
        self.params = params # 
    
        
    def reset_params_by_random(self, initseed):
        """
        params == a list of weights, like  [W1, W2, W3]
        """
        # here, we need to reset self.params
        random.seed(initseed)
        
        for W in self.params:
            (in_size, out_size) = W.shape
            scale = math.sqrt(6.0 / (in_size+out_size))*2
            bias = -math.sqrt(6.0 / (in_size+out_size))
            
            for i in range(in_size):
                for j in range(out_size):
                    W[i,j] = random.random()*scale + bias

class SimpleDSSM_0(object):
    def __init__(self, W_init_Q, W_init_D, n_mbsize, n_neg, n_shift, strategy = 0):
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
        activation = T.tanh
        self.layers_Q = []
        self.layers_D = []
        
        if strategy == 0:
            for W in W_init_Q:
                self.layers_Q.append(nc.LayerWithoutBias(W, activation))
            
            for W in W_init_D:
                self.layers_D.append(nc.LayerWithoutBias(W, activation))
                
            self.layer_cosine = nc.CosineLayer(n_mbsize, n_neg, n_shift)
    
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
 
    def dumpout(self):
        return self.params

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
#        cosine_matrix_reshape = T.reshape(cosine_matrix, (self.layer_cosine.n_mbsize, 1))
        
        return cosine_matrix

def train_dssm_with_minibatch(ps):
    # 1. Load in the input streams
    # Suppose the max seen feaid in the stream is 48930
    # then, inputstream1.nMaxFeatureId is 48931, which is one more
    # Ideally, the num_cols should be 48931
    # However, to make it conpatible with MS DSSM toolkit, we add it by one
    inputstream_src = nc.InputStream(ps.QFILE) # this will load in the whole file as origin. No modification at all
    inputstream_tgt = nc.InputStream(ps.DFILE)
    
    # 2. Load in the network structure and initial weights from DSSM
    subfields = ps.SimpleDSSM_0_NetworkStructure_src.split(':') # a line of "49k:128:128"
    weights_src = []
    for j in range(len(subfields)-1):
        in_size = int(subfields[j])
        out_size = int(subfields[j+1])
        W = np.zeros((in_size, out_size), dtype = np.float32)
        weights_src.append(W)
    
    subfields = ps.SimpleDSSM_0_NetworkStructure_tgt.split(':') # a line of "49k:128:128"
    weights_tgt = []
    for j in range(len(subfields)-1):
        in_size = int(subfields[j])
        out_size = int(subfields[j+1])
        W = np.zeros((in_size, out_size), dtype = np.float32)
        weights_tgt.append(W)
    
    
    # 3. Init two instances and do init
    # for a model, it's params = [W_list_1, W_list_2, W_list_3]
    init_model_src = SimpleDSSMModel_0_Format(weights_src)
    init_model_src.reset_params_by_random(0)
    
    init_model_tgt = SimpleDSSMModel_0_Format(weights_tgt)
    init_model_tgt.reset_params_by_random(1)

    # Before iteration, dump out the init model 
    outfilename_src = os.path.join(ps.MODELPATH, "yw_dssm_Q_0")
    outfilename_tgt = os.path.join(ps.MODELPATH, "yw_dssm_D_0")
    save_simpledssmmodel(outfilename_src, init_model_src)
    save_simpledssmmodel(outfilename_tgt, init_model_tgt)
    


    
    # 3. Generate useful index structures
    # We assue that each minibatch is of the same size, i.e. mbsize
    # if the last batch has fewer samples, just ignore it
    mbsize = inputstream_src.BatchSize
    indexes = basic_utilities.generate_index(mbsize, ps.NTRIAL, ps.SHIFT) # for a normal minibatch, we should use this indexes
#    indexes_lastone = generate_index(inputstream1.minibatches[-1].SegSize, ntrial, shift) # this is used for the last batch

    # 4. Generate an instance of DSSM    
    dssm = SimpleDSSM_0(init_model_src.params, init_model_tgt.params, mbsize, ps.NTRIAL, ps.SHIFT)

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
    
    train_output = dssm.output_train(dssm_index_Q, dssm_index_D, dssm_input_Q, dssm_input_D)
#    ywcost_scalar = ywcost.sum()
    func_train_output = theano.function([dssm_index_Q, dssm_index_D, dssm_input_Q, dssm_input_D], train_output,
                             updates=basic_utilities.gradient_updates_momentum(train_output, dssm.params, learning_rate, momentum, mbsize), mode=functionmode)
    
    iteration = 1
    while iteration <= ps.MAX_ITER:
        print "Iteration %d--------------" % (iteration)
        print "Each iteration contains %d minibatches" % (inputstream_src.nTotalBatches)
        
        trainLoss = 0.0

        if inputstream_src.BatchSize == inputstream_src.minibatches[-1].SegSize:
            usefulbatches = inputstream_src.nTotalBatches
        else:
            usefulbatches = inputstream_src.nTotalBatches -1
        print "After removing the last incomplete batch, we need to process %d batches" % (usefulbatches)

        curr_minibatch_src = np.zeros((inputstream_src.BatchSize, init_model_src.params[0].shape[0]), dtype = numpy.float32)
        curr_minibatch_tgt = np.zeros((inputstream_tgt.BatchSize, init_model_tgt.params[0].shape[0]), dtype = numpy.float32)

        # we scan all minibatches, except the last one  
        for i in range(usefulbatches):
            
            inputstream_src.setaminibatch(curr_minibatch_src, i)
            inputstream_tgt.setaminibatch(curr_minibatch_tgt, i)

            current_output = func_train_output(indexes[0], indexes[1], curr_minibatch_src, curr_minibatch_tgt)
            trainLoss += current_output

            if i%100 == 0:            
                print "%d\t%f\t%f" % (i, current_output, trainLoss)

        print "all batches in this iteraton is processed"
        print "trainLoss = %f" % (trainLoss)
                     
        # dump out current model separately
        tmpparams = []
        for W in dssm.params_Q:
            tmpparams.append(W.get_value())
        outfilename_src = os.path.join(ps.MODELPATH, "yw_dssm_Q_%d" % (iteration))
        save_simpledssmmodel(outfilename_src, SimpleDSSMModel_0_Format(tmpparams))
         

        tmpparams = []
        for W in dssm.params_D:
            tmpparams.append(W.get_value())
        outfilename_tgt = os.path.join(ps.MODELPATH, "yw_dssm_D_%d" % (iteration))
        save_simpledssmmodel(outfilename_tgt, SimpleDSSMModel_0_Format(tmpparams))
        
        print "Iteration %d-------------- is finished" % (iteration)
        
        iteration += 1

    print "-----The whole train process is finished-------\n"
    
def train_dssm_with_minibatch_predictiononly(dssm_file_src, dssm_file_tgt, validateqfile, validatedfile, validatepair, validateoutput):
    
    # 1. Load in the input streams
    inputstream_src = nc.InputStream(validateqfile) # this will load in the whole file as origin. No modification at all
    inputstream_tgt = nc.InputStream(validatedfile)
    
    # 2. Load in trained models
    init_model_src = load_simpledssmmodel(dssm_file_src)
    init_model_tgt = load_simpledssmmodel(dssm_file_tgt)

    # 3. Generate useful index structures
    # We assue that each minibatch is of the same size, i.e. mbsize
    # if the last batch has fewer samples, just ignore it
    # For prediction, we only need pairwise indexes
    mbsize = inputstream_src.BatchSize
    ntrial = 1 # dumb parameter
    shift = 1 # dumb parameter
#    indexes = generate_index(mbsize, ntrial, shift) # for a normal minibatch, we should use this indexes
    indexes = [range(mbsize), range(mbsize)]
#    indexes_lastone = generate_index(inputstream1.minibatches[-1].SegSize, ntrial, shift) # this is used for the last batch

    # 4. Generate an instance of DSSM    
    dssm = SimpleDSSM_0(init_model_src.params, init_model_tgt.params, mbsize, ntrial, shift)

    # Create Theano variables for the MLP input
    dssm_index_Q = T.ivector('dssm_index_Q')
    dssm_index_D = T.ivector('dssm_index_D')
    dssm_input_Q = T.matrix('dssm_input_Q')
    dssm_input_D = T.matrix('dssm_input_D')

    test_output = dssm.output_test(dssm_index_Q, dssm_index_D, dssm_input_Q, dssm_input_D)
    func_test_output = theano.function([dssm_index_Q, dssm_index_D, dssm_input_Q, dssm_input_D], test_output, mode=functionmode)
    
    
    print "This prediction contains totally %d minibatches" % (inputstream_src.nTotalBatches)

    curr_minibatch_src = np.zeros((inputstream_src.BatchSize, init_model_src.params[0].shape[0]), dtype = numpy.float32)
    curr_minibatch_tgt = np.zeros((inputstream_tgt.BatchSize, init_model_tgt.params[0].shape[0]), dtype = numpy.float32)

    result = []
    
    # we scan all minibatches  
    for i in range(inputstream_src.nTotalBatches):
        inputstream_src.setaminibatch(curr_minibatch_src, i)
        inputstream_tgt.setaminibatch(curr_minibatch_tgt, i)
        
        current_output = func_test_output(indexes[0], indexes[1], curr_minibatch_src, curr_minibatch_tgt)
        result.extend(current_output)
        print "batch no %d" % (i)
        

    print "all batches in this iteraton is processed"
    
    # 0. open the outputfile
    outfile = open(validateoutput, 'w')
    labelfile = open(validatepair, 'r')
    
    line_labelfile = labelfile.readline()
    line_output = line_labelfile.rstrip('\r\n') + "\tDSSM\n"
    outfile.write(line_output)
    
    for score in result:
        if math.isnan(score):
            break
        
        line_labelfile = labelfile.readline()
        line_output = "%s\t%f\n" % (line_labelfile.rstrip('\r\n'), score)
        outfile.write(line_output)
                     

    outfile.close()
    labelfile.close()
    
    
    
# input: a file name
# output: an ndarray of weights. For default DSSM, it is just a list of matrices

def load_simpledssmmodel(infilename):
    h = open(infilename, 'rb')
    model = pickle.load(h)
    h.close()
    return model

def save_simpledssmmodel(filename, model):
    h = open(filename, 'wb')
    pickle.dump(model, h)
    h.close()


def func_main(configfilename):
    print "Train SimpleDSSM_0. After each iteration, test it"
    if not os.path.exists(configfilename):
        print configfilename + " doesn't exist!"
        sys.exit(0)
    ps = basic_utilities.ParameterSetting(configfilename)

    if not os.path.exists(ps.MODELPATH):
        os.makedirs(ps.MODELPATH)

    train_dssm_with_minibatch(ps)
    
    for i in range(ps.MAX_ITER+1):
        dssm_file_src = "%s/yw_dssm_Q_%d" % (ps.MODELPATH, i)
        dssm_file_tgt = "%s/yw_dssm_D_%d" % (ps.MODELPATH, i)
        outputfilename = "%s_prediction" % (dssm_file_src)
    
        train_dssm_with_minibatch_predictiononly(dssm_file_src, dssm_file_tgt, ps.VALIDATEQFILE, ps.VALIDATEDFILE, ps.VALIDATEPAIR, outputfilename)
        
        
def func_main_testonly(configfilename):
    print "Test a specific SimpleDSSM_0 for a specific triple (Q, T, label)"
    if not os.path.exists(configfilename):
        print configfilename + " doesn't exist!"
        sys.exit(0)
    ps = basic_utilities.ParameterSetting(configfilename)

    dssm_file_src = ps.SEEDMODEL1
    dssm_file_tgt = ps.SEEDMODEL2
    outputfilename = ps.VALIDATEOUTPUT
    
    train_dssm_with_minibatch_predictiononly(dssm_file_src, dssm_file_tgt, ps.VALIDATEQFILE, ps.VALIDATEDFILE, ps.VALIDATEPAIR, outputfilename)
        
        