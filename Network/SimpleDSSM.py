"""
This network structure is the same as Microsoft DSSM.
Both Q and D go through the same network, such as 49K-128-128
A final cosine layer is applied to the embedding results.
"""


import sys
#sys.path.insert(0,'/u/yuanwei/scisoft/anaconda/lib/python2.7/site-packages')

#sys.path.append('../Utilities')  
#sys.path.append('/home/yw/workspace/test/TheanoConciseMLP/Utilities')  
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


class SimpleDSSMModelFormat(object):
    # only mlayer_num and mlink_num are integers
    # all other parameters are ndarray
    def __init__(self, mlayer_num, layer_info, mlink_num, in_num_list, out_num_list, params):
        self.mlayer_num = mlayer_num    # 3
        self.layer_info = layer_info    # array([48430,   128,   128], dtype=int32)
        self.mlink_num = mlink_num      # 2
        self.in_num_list = in_num_list  # [48430, 128]
        self.out_num_list = out_num_list    # [128, 128]
        self.params = params # a list of weights
    
    def reset_params_by_random(self, initseed):
        # here, we need to reset self.params
        random.seed(initseed)
        
        for param in self.params:
            (in_size, out_size) = param.shape
            scale = math.sqrt(6.0 / (in_size+out_size))*2
            bias = -math.sqrt(6.0 / (in_size+out_size))
            
            for i in range(in_size):
                for j in range(out_size):
                    param[i,j] = random.random()*scale + bias

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
                self.layers_Q.append(nc.LayerWithoutBias(W, activation))
            
            self.layers_D = []
            for W, activation in zip(W_init_D, activations_D):
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

    def output_embedding(self, Q, D):
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
        """
        for layer in self.layers_Q:
            QW = layer.output_linear(Q)
            QWT = layer.output(Q)
            break
        for layer in self.layers_D:
            DW = layer.output_linear(D)
            DWT = layer.output(D)
            break
        return QW, QWT, DW, DWT

        """
        QWT0 = self.layers_Q[0].output(Q)
        QE_linear = self.layers_Q[1].output_linear(QWT0)
        QE_tanh = self.layers_Q[1].output(QWT0)

        DWT0 = self.layers_D[0].output(D)
        DE_linear = self.layers_D[1].output_linear(DWT0)
        DE_tanh = self.layers_D[1].output(DWT0)
        return QE_linear, QE_tanh, DE_linear, DE_tanh                    

    def output_train_complete(self, index_Q, index_D, Q, D):
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

        return cosine_matrix, cosine_matrix_reshape, cosine_matrix_reshape_softmax, column1, column1_neglog

def train_dssm_with_minibatch(bin_file_train_1, bin_file_train_2, dssm_file_1_simple, dssm_file_2_simple, outputdir, ntrial, shift, max_iteration):
    # 1. Load in the input streams
    # Suppose the max seen feaid in the stream is 48930
    # then, inputstream1.nMaxFeatureId is 48931, which is one more
    # Ideally, the num_cols should be 48931
    # However, to make it conpatible with MS DSSM toolkit, we add it by one
    inputstream1 = nc.InputStream(bin_file_train_1) # this will load in the whole file as origin. No modification at all
    inputstream2 = nc.InputStream(bin_file_train_2)
    

    # 2. Load in the network structure and initial weights from DSSM
    init_model_1 = load_simpledssmmodel(dssm_file_1_simple)
    init_model_1.reset_params_by_random(0)
    activations_1 = [T.tanh] * init_model_1.mlink_num
    
    init_model_2 = load_simpledssmmodel(dssm_file_2_simple)
    init_model_2.reset_params_by_random(1)
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
    indexes = basic_utilities.generate_index(mbsize, ntrial, shift) # for a normal minibatch, we should use this indexes
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
                             updates=basic_utilities.gradient_updates_momentum(ywcost, dssm.params, learning_rate, momentum, mbsize), mode=functionmode)
#    ywtest_noupdate = theano.function([dssm_index_Q, dssm_index_D, dssm_input_Q, dssm_input_D], ywcost,
#                             mode=functionmode)
    # Keep track of the number of training iterations performed
#    grad_ywcost = theano.grad(ywcost, dssm.params)
#    grad_ywtest = theano.function([dssm_index_Q, dssm_index_D, dssm_input_Q, dssm_input_D], grad_ywcost, mode=functionmode)
 
#    train_embedding = dssm.output_embedding(dssm_input_Q, dssm_input_D)
#    func_train_embedding = theano.function([dssm_input_Q, dssm_input_D], train_embedding, mode=functionmode)
    
#    train_output_complete = dssm.output_train_complete(dssm_index_Q, dssm_index_D, dssm_input_Q, dssm_input_D)
#    func_train_output_complete = theano.function([dssm_index_Q, dssm_index_D, dssm_input_Q, dssm_input_D], train_output_complete, mode=functionmode)
    
    
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
#            i = 6
#            if i %100 == 0:
            
            inputstream1.setaminibatch(curr_minibatch1, i)
            inputstream2.setaminibatch(curr_minibatch2, i)

#            tmp_train_output_complete = func_train_output_complete(indexes[0], indexes[1], curr_minibatch1, curr_minibatch2)
#            tmp_train_embedding = func_train_embedding(curr_minibatch1, curr_minibatch2)
#            grad_current_output =  grad_ywtest(indexes[0], indexes[1], curr_minibatch1, curr_minibatch2)           
            current_output = ywtest(indexes[0], indexes[1], curr_minibatch1, curr_minibatch2)
#            current_output = ywtest_noupdate(indexes[0], indexes[1], curr_minibatch1, curr_minibatch2)
#            print "batch no %d, %f\n" % (i, current_output)
#           print current_output
            trainLoss += current_output
#            print "After processing batch no %d, curr_loss = %f, trainLoss = %f" % (i, current_output, trainLoss)
            if i%100 == 0:            
                print "%d\t%f\t%f" % (i, current_output, trainLoss)

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
    
def train_dssm_with_minibatch_predictiononly(bin_file_test_1, bin_file_test_2, dssm_file_1_simple, dssm_file_2_simple, labelfilename, outputfilename):
    
    # 0. open the outputfile
    outfile = open(outputfilename, 'w')
    labelfile = open(labelfilename, 'r')
    
    # 1. Load in the input streams
    # Suppose the max seen feaid in the stream is 48930
    # then, inputstream1.nMaxFeatureId is 48931, which is one more
    # Ideally, the num_cols should be 48931
    # However, to make it conpatible with MS DSSM toolkit, we add it by one
    inputstream1 = nc.InputStream(bin_file_test_1) # this will load in the whole file as origin. No modification at all
    inputstream2 = nc.InputStream(bin_file_test_2)
    
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


def func_main(configfilename):
    print "We need to train a simple dssm model from the beginning"
    if not os.path.exists(configfilename):
        print configfilename + " doesn't exist!"
        sys.exit(0)
    ps = basic_utilities.ParameterSetting(configfilename)

    if not os.path.exists(ps.outputdir):
        os.makedirs(ps.outputdir)

    train_dssm_with_minibatch(ps.bin_file_train_1, ps.bin_file_train_2, ps.dssm_file_1_simple, ps.dssm_file_2_simple, ps.outputdir, ps.ntrial, ps.shift, ps.max_iteration)    

    for i in range(ps.max_iteration+1):
        dssm_file_1_predict = "%s/yw_dssm_Q_%d" % (ps.outputdir, i)
        dssm_file_2_predict = "%s/yw_dssm_D_%d" % (ps.outputdir, i)
        outputfilename = "%s_prediction" % (dssm_file_1_predict)
    
        train_dssm_with_minibatch_predictiononly(ps.bin_file_test_1, ps.bin_file_test_2, dssm_file_1_predict, dssm_file_2_predict, ps.labelfile, outputfilename)
        
def func_main_testonly(configfilename):
    print "We need to evaluate a SimpleDSSM_1 model on some test data"
    if not os.path.exists(configfilename):
        print configfilename + " doesn't exist!"
        sys.exit(0)
    ps = basic_utilities.ParameterSetting(configfilename)

    dssm_file_src_predict = ps.dssm_file_1_simple
    dssm_file_tgt_predict = ps.dssm_file_2_simple
    outputfilename = "%s_testonly" % (dssm_file_src_predict)
    
    train_dssm_with_minibatch_predictiononly(ps.bin_file_test_1, ps.bin_file_test_2, dssm_file_src_predict, dssm_file_tgt_predict, ps.labelfile, outputfilename)
        
        