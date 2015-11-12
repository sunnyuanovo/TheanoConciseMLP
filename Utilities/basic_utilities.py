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

class ParameterSetting(object):
    def __init__(self, configfilename):
        # parameters should be consistent with the dssm config file
        self.shift = 1
        
        f = open(configfilename, "r")
        for line in f:
            fields =  line.rstrip('\r\n').split('\t') # remove any trailing combinaiton of '\r' or '\n'
            if len(fields) < 2:
                continue
            
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
            elif fields[0] == "VALIDATEOUTPUT":
                self.validateoutput = fields[1]
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
            
            # below are parameters fro SimpleDSSM_1
            elif fields[0] == "QFILE_1":
                self.bin_file_train_src_1 = fields[1]
                continue
            elif fields[0] == "QFILE_2":
                self.bin_file_train_src_2 = fields[1]
                continue
            elif fields[0] == "DFILE_1":
                self.bin_file_train_tgt_1 = fields[1]
                continue
            elif fields[0] == "DFILE_2":
                self.bin_file_train_tgt_2 = fields[1]
                continue
            elif fields[0] == "VALIDATEQFILE_1":
                self.bin_file_test_src_1 = fields[1]
                continue
            elif fields[0] == "VALIDATEQFILE_2":
                self.bin_file_test_src_2 = fields[1]
                continue
            elif fields[0] == "VALIDATEDFILE_1":
                self.bin_file_test_tgt_1 = fields[1]
                continue
            elif fields[0] == "VALIDATEDFILE_2":
                self.bin_file_test_tgt_2 = fields[1]
                continue
            elif fields[0] == "SimpleDSSM_1_NetworkStructure_src":
                self.SimpleDSSM_1_NetworkStructure_src = fields[1]
            elif fields[0] == "SimpleDSSM_1_NetworkStructure_tgt":
                self.SimpleDSSM_1_NetworkStructure_tgt = fields[1]
        f.close()


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


