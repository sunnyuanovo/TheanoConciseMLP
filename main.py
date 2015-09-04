import numpy as np
import numpy
import matplotlib.pyplot as plt
import theano
# By convention, the tensor submodule is loaded as T
import theano.tensor as T

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
            - q_ind, d_ind : int
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
    def __init__(self, W_init, b_init, n_mbsize, n_neg, n_shift,  activations):
        '''
        This class is similar to MLP, except that we need to construct separate models for Q and D, 
        then add a cosine label at the end
        
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
        self.layers_Q = []
        self.layers_D = []
        # Construct the layers
        for W, b, activation in zip(W_init, b_init, activations):
            self.layers_Q.append(Layer(W, b, activation))
            self.layers_D.append(Layer(W, b, activation))
        
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
                            updates=gradient_updates_momentum(cost, mlp.params, learning_rate, momentum), mode='DebugMode')
    # Create a theano function for computing the MLP's output given some input
    mlp_output = theano.function([mlp_input], mlp.output(mlp_input), mode='DebugMode')
    
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
                            updates=gradient_updates_momentum(cost, dssm.params, learning_rate, momentum), mode='DebugMode')
    # Create a theano function for computing the MLP's output given some input
    dssm_output = theano.function([dssm_index_Q, dssm_index_D, dssm_input_Q, dssm_input_D], cost_test, mode='DebugMode')
    
    
    ywcost = dssm.output_train_test(dssm_index_Q, dssm_index_D, dssm_input_Q, dssm_input_D)
    ywtest = theano.function([dssm_index_Q, dssm_index_D, dssm_input_Q, dssm_input_D], ywcost, mode='DebugMode')
    
    # Keep track of the number of training iterations performed
    iteration = 0
    max_iteration = 1
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
        print "indexes[0] = ", indexes[0]
        print "indexes[1] = ", indexes[1]
        
        current_output = ywtest(indexes[0], indexes[1], X, X1)
#        current_output = ywtest(X, X1)
        print current_output
  
              
        
        
        
        # We can compute the accuracy by thresholding the output
        # and computing the proportion of points whose class match the ground truth class.
        
        iteration += 1

if __name__ == '__main__':
    test_dssm()
    print '----------------finished--------------------'
