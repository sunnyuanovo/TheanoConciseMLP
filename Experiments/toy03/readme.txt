Here, I just the following to generate dssm models

BATCHSIZE	2
NTRIAL	1
MAX_ITER	5
PARM_GAMMA	10
TRAIN_TEST_RATE	1
LEARNINGRATE	0.1
SOURCE_LAYER_DIM	2

The key point is to understand that given (Q,D) pairs and a given dssm model, the forward prediction (i.e. cosine(Q,D) is correct). 
The test is OK

Also, suppose that everything is row based.
input_data.shape = (mbsize, 5)
weight.shape = (5, 2)
When saving this weight, it's row based. The 1st row is output first, then the 2nd row, and then 3rd...


