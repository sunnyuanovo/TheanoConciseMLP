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
