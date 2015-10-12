import sys
sys.path.insert(0,'/u/yuanwei/scisoft/anaconda/lib/python2.7/site-packages')

sys.path.append('./Utilities')  
sys.path.append('/home/yw/workspace/test/TheanoConciseMLP/Utilities')  
import basic_utilities
import test_utilities

sys.path.append('./Network')  
sys.path.append('/home/yw/workspace/test/TheanoConciseMLP/Network')
import SimpleDSSM

#from SimpleDSSM import *



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



if __name__ == '__main__':
    
    if len(sys.argv) >=3:
#        print 'Argument List', str(sys.argv)
        if sys.argv[1] == "-convertmicrosoftdssmmodel":
            assert(len(sys.argv) == 4)
            print "We need to convert a dssm model file from Microsot Format to Simple Format"
            SimpleDSSM.convert_microsoftdssmmodel(sys.argv[2], sys.argv[3])
        elif sys.argv[1] == "-train":
            SimpleDSSM.func_main(sys.argv[2])
        
    else:
        print 'Error\n'
    sys.exit(0)

    print '----------------finished--------------------'
