# TheanoConciseMLP

1. Dataset
All data files are under the folder "Dataset/toy03". The ".fea" files are raw data. 
For example, "train.1.src.seq.fea" is the features extracted from source, i.e. Query. It's content is as follows:
0:1 1:1           //(f0 f1)
2:1 3:1 4:1       //(f2 f3 f4)
1:1 3:1           //(f1 f3)
1:1 2:1 3:1 4:1   //(f1 f2 f3 f4)
The meaning of each line is appended after "//". We don't need to worry about the exact meaning of the features.
All we need to know is that we have totally 4 queries, and the feature space is [f0, f1, f2, f3, f4]
The ".bin" files are just binary format of corresponding ".fea" files, with auxiliary information.
Don't worry about the details of ".bin" files. My code will read the "bin" files and recover the raw information.
After loading "train.1.src.seq.bin" file, we will have the following matrix as Q in memory:
1 1 0 0 0
0 0 1 1 1
0 1 0 1 0
0 1 1 1 1
Here, each row represents a query vector

In addition, we use a minibatch size = 2. This is also encoded in the "bin" file.

2. Model Structure
Given a query (or doc) vector of length 5, we map it to a vector of length 2. Therefore, for Query/Doc part, 
we have a weight martix with shape (5,2) respectively.
After this embedding, we will be able to compute cosine between them, compute an objective function, and then 
update weights using gradient descendent. For details, please check "http://research.microsoft.com/en-us/um/people/jfgao/paper/2013/cikm2013_DSSM_fullversion.pdf"

Or we can disccus over phone

3. Initial Models
You could find them under "/Experiments/toy03/WebSearch/config_WebSearch_FullyConnect.txt.train"
All the files with patten "DSSM_QUERY_ITER?_simple" and "DSSM_DOC_ITER?_simple" are models obtained using Microsoft
toolkit.

4. Verification Process (Forward Propogation)
Given an initial model pair of (Q, D), I conduct forward propogation and print out the final cosine information.
This step is verified.

5. Verification Process (Backward Propogation)
This step fails. Given the following initial model pair
"DSSM_QUERY_ITER0_simple" and "DSSM_DOC_ITER0_simple"
We first load in the 1st batch from Q and D. The forward propogation is correct. But after that, back propogation 
is incorrect. Therefore, the weights are not updated correctly.











