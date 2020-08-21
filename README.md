# HW_ML
Machine Learning (university course, 6th semester) homework

### Lab1 - Regression
lab1/lab1.ipynb
* Select variables using correlation
* OLS - ordinary least squares
* Plot the residuals for OLS
* Ridge regression
* PCA - principal components analysis

lab1/speed_test_1.py
* Generate variables and create a response with dimensionality as a parameter
* OLS from scratch (matrix solution)
* OLS as library function in statsmodels and scikit-learn
* Run speed test on OLS for different dimensions of variables and compare the results

benchmark/benchmark_cls.py - benchmark package
* Decorator class Benchmark which can get the runtime without loosing returned by the function values

### Lab2 - Classification. Part 1
Naive Bayes, k-nn, logistic regression

lab2/lab22.ipynb
* Implement summary function, which returns accuracy, precision, fall-out and plots confusion matrices for both training and test sets
* Naive Bayes classifier with different distributions assumed for the likelihood of the features
* K-nearest neighbors for $k \in \overline{2,12}$
* Logistic regression - using the class, implemented from scratch
* Dealing with factor variables - split the data into 3 groups and perform classification on them separately

logit/logit_cls.py
* Class Logit - performs logistic regression with mini-batch gradient descent

logit/logit_functional.py
* Procedural version of logistic regression classifier; uses batch gradient descent

lab2/logit_scratch_test.py
* Test if logistic regression classifiers implemented from scratch work properly

### Lab3 - Classification. Part 2
SVM, tree-based algorithms

lab3/lab3.ipynb
* Copy-paste summary function from previous lab
* Implement decision boundary visualization function for SVM and tree-based algorithms
* SVM - support vector machine classifier with linear, polynomial, exponential and tanh kernels
* Decision tree classification with different number of samples on each leaf
* Visualize decision tree as a graph using graphviz library
* Boosting algorithms - AdaBoost, Gradient Boosting; perform parameter search for best learning rate and estimators quantity
* Random forest classification; we use the grid search to choose how many estimators do we need here, too
* We once again deal with factor variables by splitting the data into 3 groups and running on them all the algorithms above

### Lab4 - Collaborative Filtering
PMF - probabilistic matrix factorization. It's a collaborative filtering method 
together with NMF - non-negative matrix factorization
Here we will use it on last.fm dataset containing information 
about users and how often do they listen to different artists' songs

lab4/lab4.ipynb
* Prepare data for filtering - form the score with ln(the time user spent on listening to the artists' songs)
* The data looks like log-normal, but in fact, it is not; by the method in the [paper](https://papers.nips.cc/paper/3208-probabilistic-matrix-factorization.pdf "PMF original paper") we assume the normality anyway. 
  Perhaps it would've been better to find out which distribution we were working with and then generate it in the PMF algorithm...
  I just haven't got enough time for that during the semester :(
* Run PMF on the prepared data - user-score matrix. This is kind of demo, main train cycle is located in the next file

PMF/pmf.py, lab4/l4.py
* PMF algorithm implementation with speed optimization. 
  The first file was updated after a successful training cycle in the second file, 
  so technically PMF functions are the same here

lab4/lab4_predict.py
* Demonstrate "predict vs true" on a few samples

### Lab5 - Neural Networks
The task was to implement a deep neural network with fully-connected layers
for binary classification, 
forward and backward propagation algorithms from scratch

NN/nn_core.py
* Implementation of NNBinaryClassifier class with forward and backward propagation 
  algorithms encapsulated in the "fit" method
* Available methods: fit, predict, save (save current weights) and decision_boundary (visualize decision boundary in 2D)

lab5/laba.ipynb
* Visualize toy classes in 2D
* Train and test neural network, check if it overfits the data
* Visualize NN's decision boundary
