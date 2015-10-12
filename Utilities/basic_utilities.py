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



