"""
This network structure is a improved version of Microsoft DSSM.
For Q, suppose we have 2 representations: l3g and word, l3g and word2vec
l3g-->l3g_128
word-->word_128
cat(l3g_128 and word_128) -->final_128
A final cosine layer is applied to the final embedding results.
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

class SimpleDSSM1Model_1_Format(object):
    def __init__(self, params):
        """
        params == a list of list, each elem is a matrix [ [W1, W2, W3], [W4, w5], [W6, W7]]
        """
        self.params = params # 
    
        
    def reset_params_by_random(self, initseed):
        """
        self.params == a list of list, each elem is a matrix [ [W1, W2, W3], [W4, w5], [W6, W7]]
        """
        # here, we need to reset self.params
        random.seed(initseed)
        
        for param in self.params:
            for W in param:
                (in_size, out_size) = W.shape
                scale = math.sqrt(6.0 / (in_size+out_size))*2
                bias = -math.sqrt(6.0 / (in_size+out_size))
                
                for i in range(in_size):
                    for j in range(out_size):
                        W[i,j] = random.random()*scale + bias
        
class SimpleDSSM_1(object):
    def __init__(self, W_init_Q, W_init_D, n_mbsize, n_neg, n_shift, strategy = 0):
        """
        W_init_Q = a list of list, each elem is a matrix [ [W1, W2, W3], [W4, w5], [W6, W7]]
        W_init_Q = [[weights list for l3g], [weight list for word ], [weight list for merge]]
        [[49k,128] for l3g, [100k,128] for word, [256, 128] for merge] 
        """
        activation = T.tanh
        """
        self.layers_Q_1 = []
        self.layers_Q_2 = []
        self.layers_Q_merge = []
        self.layers_D_1 = []
        self.layers_D_2 = []
        self.layers_D_merge = []
        """
        self.layers_Q = [[], [], []]
        self.layers_D = [[], [], []]
        
        if strategy == 0:
            # Init stuff for Q
            """
            for W in W_init_Q[0]:
    #            self.layers_Q_1.append(nc.LayerWithoutBias(W, activation))
                self.layers_Q[0].append(nc.LayerWithoutBias(W, activation))
            for W in W_init_Q[1]:
                self.layers_Q_2.append(nc.LayerWithoutBias(W, activation))
            for W in W_init_Q[2]:
                self.layers_Q_merge.append(nc.LayerWithoutBias(W, activation))

            # Init stuff for D
            for W in W_init_D[0]:
                self.layers_D_1.append(nc.LayerWithoutBias(W, activation))
            for W in W_init_D[1]:
                self.layers_D_2.append(nc.LayerWithoutBias(W, activation))
            for W in W_init_D[2]:
                self.layers_D_merge.append(nc.LayerWithoutBias(W, activation))

            """
            for i in range(len(W_init_Q)):
                for W in W_init_Q[i]:
                    self.layers_Q[i].append(nc.LayerWithoutBias(W, activation))
            for i in range(len(W_init_D)):
                for W in W_init_D[i]:
                    self.layers_D[i].append(nc.LayerWithoutBias(W, activation))
                    
            self.layer_cosine = nc.CosineLayer(n_mbsize, n_neg, n_shift)
    
            # Combine parameters from all layers
            self.params = []
            self.params_Q = [[], [], []]
            self.params_D = [[], [], []]

            """            
            for layer in self.layers_Q_1:
                self.params += layer.params
                self.params_Q += layer.params
            for layer in self.layers_Q_2:
                self.params += layer.params
                self.params_Q += layer.params
            for layer in self.layers_Q_merge:
                self.params += layer.params
                self.params_Q += layer.params
                
            for layer in self.layers_D_1:
                self.params += layer.params
                self.params_D += layer.params
            for layer in self.layers_D_2:
                self.params += layer.params
                self.params_D += layer.params
            for layer in self.layers_D_merge:
                self.params += layer.params
                self.params_D += layer.params
            """

            for i in range(len(self.layers_Q)):
                for j in range(len(self.layers_Q[i])):
                    self.params += self.layers_Q[i][j].params
                    self.params_Q[i] += self.layers_Q[i][j].params
                    
            for i in range(len(self.layers_D)):
                for j in range(len(self.layers_D[i])):
                    self.params += self.layers_D[i][j].params
                    self.params_D[i] += self.layers_D[i][j].params

    def dumpout(self):
        return self.params

    def output_train(self, index_Q, index_D, Q_1, Q_2, D_1, D_2):
        for layer in self.layers_Q_1:
            Q_1 = layer.output(Q_1)
        
        for layer in self.layers_Q_2:
            Q_2 = layer.output(Q_2)

        Q_m = T.concatenate([Q_1, Q_2], axis = 1) # Q_merge
        
        for layer in self.layers_Q_merge:
            Q_m = layer.output(Q_m)
            

        for layer in self.layers_D_1:
            D_1 = layer.output(D_1)
        
        for layer in self.layers_D_2:
            D_2 = layer.output(D_2)

        D_m = T.concatenate([D_1, D_2], axis = 1) # D_merge
        
        for layer in self.layers_D_merge:
            D_m = layer.output(D_m)

        
        cosine_matrix = self.layer_cosine.output_noloop_1(index_Q, index_D, Q_m, D_m) * 10 # scaling by 10 is suggested by Jianfeng Gao
    
        cosine_matrix_reshape = T.reshape(cosine_matrix, (self.layer_cosine.n_mbsize, self.layer_cosine.n_neg+1))
        
        cosine_matrix_reshape_softmax = T.nnet.softmax(cosine_matrix_reshape)

        column1 = cosine_matrix_reshape_softmax[:,0]
        
        column1_neglog = -T.log(column1)

        return column1_neglog.sum()

    def output_test(self, index_Q, index_D, Q_1, Q_2, D_1, D_2):
        for layer in self.layers_Q_1:
            Q_1 = layer.output(Q_1)
        
        for layer in self.layers_Q_2:
            Q_2 = layer.output(Q_2)

        Q_m = T.concatenate([Q_1, Q_2], axis = 1)
        
        for layer in self.layers_Q_merge:
            Q_m = layer.output(Q_m)
            

        for layer in self.layers_D_1:
            D_1 = layer.output(D_1)
        
        for layer in self.layers_D_2:
            D_2 = layer.output(D_2)

        D_m = T.concatenate([D_1, D_2], axis = 1)
        
        for layer in self.layers_D_merge:
            D_m = layer.output(D_m)
        
        cosine_matrix = self.layer_cosine.output_noloop_1(index_Q, index_D, Q_m, D_m)
        cosine_matrix_reshape = T.reshape(cosine_matrix, (self.layer_cosine.n_mbsize, 1))
        
        return cosine_matrix_reshape

    def output_train_1(self, index_Q, index_D, Q_1, Q_2, D_1, D_2):
        for layer in self.layers_Q[0]:
            Q_1 = layer.output(Q_1)
        
        for layer in self.layers_Q[1]:
            Q_2 = layer.output(Q_2)

        Q_m = T.concatenate([Q_1, Q_2], axis = 1) # Q_merge
        
        for layer in self.layers_Q[2]:
            Q_m = layer.output(Q_m)
            

        for layer in self.layers_D[0]:
            D_1 = layer.output(D_1)
        
        for layer in self.layers_D[1]:
            D_2 = layer.output(D_2)

        D_m = T.concatenate([D_1, D_2], axis = 1) # D_merge
        
        for layer in self.layers_D[2]:
            D_m = layer.output(D_m)

        
        cosine_matrix = self.layer_cosine.output_noloop_1(index_Q, index_D, Q_m, D_m) * 10 # scaling by 10 is suggested by Jianfeng Gao
    
        cosine_matrix_reshape = T.reshape(cosine_matrix, (self.layer_cosine.n_mbsize, self.layer_cosine.n_neg+1))
        
        cosine_matrix_reshape_softmax = T.nnet.softmax(cosine_matrix_reshape)

        column1 = cosine_matrix_reshape_softmax[:,0]
        
        column1_neglog = -T.log(column1)

        return column1_neglog.sum()

    def output_test_1(self, index_Q, index_D, Q_1, Q_2, D_1, D_2):
        for layer in self.layers_Q[0]:
            Q_1 = layer.output(Q_1)
        
        for layer in self.layers_Q[1]:
            Q_2 = layer.output(Q_2)

        Q_m = T.concatenate([Q_1, Q_2], axis = 1) # Q_merge
        
        for layer in self.layers_Q[2]:
            Q_m = layer.output(Q_m)
            

        for layer in self.layers_D[0]:
            D_1 = layer.output(D_1)
        
        for layer in self.layers_D[1]:
            D_2 = layer.output(D_2)

        D_m = T.concatenate([D_1, D_2], axis = 1) # D_merge
        
        for layer in self.layers_D[2]:
            D_m = layer.output(D_m)
        
        cosine_matrix = self.layer_cosine.output_noloop_1(index_Q, index_D, Q_m, D_m)
#        cosine_matrix_reshape = T.reshape(cosine_matrix, (self.layer_cosine.n_mbsize, 1))
        
        return cosine_matrix

def func_test1():
    X = np.zeros((4,2))
    Y = np.ones((4,2))
    Z = Y*2
    weightQ = [[X], [Y], [Z]]
    dssm = SimpleDSSM_1(weightQ, weightQ, 2, 1, 1)
    result = dssm.dumpout()
    result = 0

def train_dssm_with_minibatch(ps):
    # 1. Load in the input streams of queries
    inputstream_src_1 = nc.InputStream(ps.bin_file_train_src_1)
    inputstream_src_2 = nc.InputStream(ps.bin_file_train_src_2)

    inputstream_tgt_1 = nc.InputStream(ps.bin_file_train_tgt_1)
    inputstream_tgt_2 = nc.InputStream(ps.bin_file_train_tgt_2)

    # 2. Read in network structure details and then do initialization for queries
    fields = ps.SimpleDSSM_1_NetworkStructure_src.split() # a line of "49k:128:128 100K:128:128 256:128"
    assert(len(fields) == 3)
    weights_src = [[], [], []]
    for i in range(len(fields)):
        subfields = fields[i].split(':')
        for j in range(len(subfields)-1):
            in_size = int(subfields[j])
            out_size = int(subfields[j+1])
            W = np.zeros((in_size, out_size), dtype = np.float32)
            weights_src[i].append(W)
    
    fields = ps.SimpleDSSM_1_NetworkStructure_tgt.split() # a line of "49k:128:128 100K:128:128 256:128"
    assert(len(fields) == 3)
    weights_tgt = [[], [], []]
    for i in range(len(fields)):
        subfields = fields[i].split(':')
        for j in range(len(subfields)-1):
            in_size = int(subfields[j])
            out_size = int(subfields[j+1])
            W = np.zeros((in_size, out_size), dtype = np.float32)
            weights_tgt[i].append(W)
    
    
    # 3. Init two instances and do init
    # for a model, it's params = [W_list_1, W_list_2, W_list_3]
    init_model_src = SimpleDSSM1Model_1_Format(weights_src)
    init_model_src.reset_params_by_random(0)
    
    init_model_tgt = SimpleDSSM1Model_1_Format(weights_tgt)
    init_model_tgt.reset_params_by_random(1)

    # 4. Before iteration, dump out the init model 
    # the essential part of a model is [W_list_1, W_list_2, W_list_3]
    outfilename_src = os.path.join(ps.outputdir, "yw_dssm_Q_0")
    outfilename_tgt = os.path.join(ps.outputdir, "yw_dssm_D_0")
    save_simpledssmmodel(outfilename_src, init_model_src)
    save_simpledssmmodel(outfilename_tgt, init_model_tgt)
    

    # 5. Generate useful index structures
    mbsize = inputstream_src_1.BatchSize
    indexes = basic_utilities.generate_index(mbsize, ps.ntrial, ps.shift) # for a normal minibatch, we should use this indexes

    # 6. Generate an instance of DSSM    
    # init_model_src.params ==== a list of list of weight matrices
    dssm = SimpleDSSM_1(init_model_src.params, init_model_tgt.params, mbsize, ps.ntrial, ps.shift )

    # 7. Create Theano variables for the MLP input
    dssm_index_Q = T.ivector('dssm_index_Q')
    dssm_index_D = T.ivector('dssm_index_D')

    dssm_input_Q_1 = T.matrix('dssm_input_Q_1')
    dssm_input_Q_2 = T.matrix('dssm_input_Q_2')
    dssm_input_D_1 = T.matrix('dssm_input_D_1')
    dssm_input_D_2 = T.matrix('dssm_input_D_2')

    learning_rate = 0.1
    momentum = 0.0
    
    train_output = dssm.output_train_1(dssm_index_Q, dssm_index_D, dssm_input_Q_1, dssm_input_Q_2, dssm_input_D_1, dssm_input_D_2)
    func_train_output = theano.function([dssm_index_Q, dssm_index_D, dssm_input_Q_1, dssm_input_Q_2, dssm_input_D_1, dssm_input_D_2], train_output,
                             updates=basic_utilities.gradient_updates_momentum(train_output, dssm.params, learning_rate, momentum, mbsize), mode=functionmode)
    
    iteration = 1
    while iteration <= ps.max_iteration:
        print "Iteration %d--------------" % (iteration)
        print "Each iteration contains %d minibatches" % (inputstream_src_1.nTotalBatches)
        
        trainLoss = 0.0

        if inputstream_src_1.BatchSize == inputstream_src_1.minibatches[-1].SegSize:
            usefulbatches = inputstream_src_1.nTotalBatches
        else:
            usefulbatches = inputstream_src_1.nTotalBatches -1
        print "After removing the last incomplete batch (if there is one), we need to process %d batches" % (usefulbatches)

#        usefulbatches = 10
        
        curr_minibatch_src_1 = np.zeros((mbsize, init_model_src.params[0][0].shape[0]), dtype = numpy.float32)
        curr_minibatch_src_2 = np.zeros((mbsize, init_model_src.params[1][0].shape[0]), dtype = numpy.float32)
        
        curr_minibatch_tgt_1 = np.zeros((mbsize, init_model_tgt.params[0][0].shape[0]), dtype = numpy.float32)
        curr_minibatch_tgt_2 = np.zeros((mbsize, init_model_tgt.params[1][0].shape[0]), dtype = numpy.float32)
        
        for i in range(usefulbatches):
            
            inputstream_src_1.setaminibatch(curr_minibatch_src_1, i)
            inputstream_src_2.setaminibatch(curr_minibatch_src_2, i)
            inputstream_tgt_1.setaminibatch(curr_minibatch_tgt_1, i)
            inputstream_tgt_2.setaminibatch(curr_minibatch_tgt_2, i)
            
            current_output = func_train_output(indexes[0], indexes[1], curr_minibatch_src_1, curr_minibatch_src_2, curr_minibatch_tgt_1, curr_minibatch_tgt_2)
            trainLoss += current_output
            
            if i %100 == 0:
                print "%d\t%f\t%f" % (i, current_output, trainLoss)

        print "all batches in this iteraton is processed"
        print "trainLoss = %f" % (trainLoss)
                     
        # dump out current model separately
        tmpparams = [[], [], []]
        for i in range(len(dssm.params_Q)):
            list_len = len(dssm.params_Q[i])
            for j in range(list_len):
                tmpparams[i].append(dssm.params_Q[i][j].get_value())
            
        outfilename_src = os.path.join(ps.outputdir, "yw_dssm_Q_%d" % (iteration))
        save_simpledssmmodel(outfilename_src, SimpleDSSM1Model_1_Format(tmpparams))
        

        tmpparams = [[], [], []]
        for i in range(len(dssm.params_D)):
            list_len = len(dssm.params_D[i])
            for j in range(list_len):
                tmpparams[i].append(dssm.params_D[i][j].get_value())
        outfilename_tgt = os.path.join(ps.outputdir, "yw_dssm_D_%d" % (iteration))
        save_simpledssmmodel(outfilename_tgt, SimpleDSSM1Model_1_Format(tmpparams))
        
        print "Iteration %d-------------- is finished" % (iteration)
        
        iteration += 1

    print "-----The whole train process is finished-------\n"

def train_dssm_with_minibatch_predictiononly(ps,  dssm_file_src_predict, dssm_file_tgt_predict, outputfilename):
    
    # 1. Load in the input streams of queries
    inputstream_src_1 = nc.InputStream(ps.bin_file_test_src_1)
    inputstream_src_2 = nc.InputStream(ps.bin_file_test_src_2)

    inputstream_tgt_1 = nc.InputStream(ps.bin_file_test_tgt_1)
    inputstream_tgt_2 = nc.InputStream(ps.bin_file_test_tgt_2)

    # 2. Load in existing models
    init_model_src = load_simpledssmmodel(dssm_file_src_predict)
    init_model_tgt = load_simpledssmmodel(dssm_file_tgt_predict)
    
    # 5. Generate useful index structures
    mbsize = inputstream_src_1.BatchSize
    indexes = [range(mbsize), range(mbsize)]

    # 6. Generate an instance of DSSM    
    # init_model_src.params ==== a list of list of weight matrices
    dssm = SimpleDSSM_1(init_model_src.params, init_model_tgt.params, mbsize, 1, 1 )

    # 7. Create Theano variables for the MLP input
    dssm_index_Q = T.ivector('dssm_index_Q')
    dssm_index_D = T.ivector('dssm_index_D')

    dssm_input_Q_1 = T.matrix('dssm_input_Q_1')
    dssm_input_Q_2 = T.matrix('dssm_input_Q_2')
    dssm_input_D_1 = T.matrix('dssm_input_D_1')
    dssm_input_D_2 = T.matrix('dssm_input_D_2')

    test_output = dssm.output_test_1(dssm_index_Q, dssm_index_D, dssm_input_Q_1, dssm_input_Q_2, dssm_input_D_1, dssm_input_D_2)
    func_test_output = theano.function([dssm_index_Q, dssm_index_D, dssm_input_Q_1, dssm_input_Q_2, dssm_input_D_1, dssm_input_D_2], test_output,
                            mode=functionmode)
    
    result = []
    
    curr_minibatch_src_1 = np.zeros((mbsize, init_model_src.params[0][0].shape[0]), dtype = numpy.float32)
    curr_minibatch_src_2 = np.zeros((mbsize, init_model_src.params[1][0].shape[0]), dtype = numpy.float32)
    
    curr_minibatch_tgt_1 = np.zeros((mbsize, init_model_tgt.params[0][0].shape[0]), dtype = numpy.float32)
    curr_minibatch_tgt_2 = np.zeros((mbsize, init_model_tgt.params[1][0].shape[0]), dtype = numpy.float32)
    
    for i in range(inputstream_src_1.nTotalBatches):
        
        inputstream_src_1.setaminibatch(curr_minibatch_src_1, i)
        inputstream_src_2.setaminibatch(curr_minibatch_src_2, i)
        inputstream_tgt_1.setaminibatch(curr_minibatch_tgt_1, i)
        inputstream_tgt_2.setaminibatch(curr_minibatch_tgt_2, i)
        
        current_output = func_test_output(indexes[0], indexes[1], curr_minibatch_src_1, curr_minibatch_src_2, curr_minibatch_tgt_1, curr_minibatch_tgt_2)
        result.extend(current_output)

    print "all batches in this iteraton is processed"
                     
    # 0. Open the label file and outputfile
    outfile = open(outputfilename, 'w')
    labelfile = open(ps.labelfile, 'r')

    line_labelfile = labelfile.readline()
    line_output = line_labelfile.rstrip('\r\n') + "\tDSSM\n"
    outfile.write(line_output)
    
    for score in result:
        if math.isnan(score):
            break
        
        line_labelfile = labelfile.readline()
        line_output = "%s\t%f\n" % (line_labelfile.rstrip('\r\n'), score)
#        outfile.write(str(score))
        outfile.write(line_output)
                     

    outfile.close()
    labelfile.close()
 
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
    print "We need to train a SimpleDSSM_1 model from the beginning"
    if not os.path.exists(configfilename):
        print configfilename + " doesn't exist!"
        sys.exit(0)
    ps = basic_utilities.ParameterSetting(configfilename)

    if not os.path.exists(ps.outputdir):
        os.makedirs(ps.outputdir)

    train_dssm_with_minibatch(ps)

    for i in range(ps.max_iteration+1):
        dssm_file_src_predict = "%s/yw_dssm_Q_%d" % (ps.outputdir, i)
        dssm_file_tgt_predict = "%s/yw_dssm_D_%d" % (ps.outputdir, i)
        outputfilename = "%s_prediction" % (dssm_file_src_predict)
    
        train_dssm_with_minibatch_predictiononly(ps, dssm_file_src_predict, dssm_file_tgt_predict, outputfilename)

def func_main_testonly(configfilename):
    print "We need to evaluate a SimpleDSSM_1 model on some test data"
    if not os.path.exists(configfilename):
        print configfilename + " doesn't exist!"
        sys.exit(0)
    ps = basic_utilities.ParameterSetting(configfilename)

    dssm_file_src_predict = ps.dssm_file_1_simple
    dssm_file_tgt_predict = ps.dssm_file_2_simple
    outputfilename = "%s_testonly" % (dssm_file_src_predict)
    
    train_dssm_with_minibatch_predictiononly(ps, dssm_file_src_predict, dssm_file_tgt_predict, outputfilename)
