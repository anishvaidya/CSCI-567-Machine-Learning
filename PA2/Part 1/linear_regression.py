"""
Do not change the input and output format.
If our script cannot run your code or the format is improper, your code will not be graded.

The only functions you need to implement in this template is linear_regression_noreg, linear_regression_invertible，regularized_linear_regression,
tune_lambda, test_error and mapping_data.
"""

import numpy as np
import pandas as pd

###### Q1.1 ######
def mean_absolute_error(w, X, y):
    """
    Compute the mean absolute error on test set given X, y, and model parameter w.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing test feature.
    - y: A numpy array of shape (num_samples, ) containing test label
    - w: a numpy array of shape (D, )
    Returns:
    - err: the mean absolute error
    """
    #####################################################
    # TODO 1: Fill in your code here #
    #####################################################
    y_dash = np.dot(w, X.T)
    err = np.sum(np.absolute(y_dash - y)) / X.shape[0]
#    print (type(err))
    return err

###### Q1.2 ######
def linear_regression_noreg(X, y):
  """
  Compute the weight parameter given X and y.
  Inputs:
  - X: A numpy array of shape (num_samples, D) containing feature.
  - y: A numpy array of shape (num_samples, ) containing label
  Returns:
  - w: a numpy array of shape (D, )
  """
  #####################################################
  #	TODO 2: Fill in your code here #
  #####################################################		
  w = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y)
  return w

###### Q1.3 ######
def linear_regression_invertible(X, y):
    """
    Compute the weight parameter given X and y.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing feature.
    - y: A numpy array of shape (num_samples, ) containing label
    Returns:
    - w: a numpy array of shape (D, )
    """
    #####################################################
    # TODO 3: Fill in your code here #
    #####################################################
#    if (X_evalue_abs < (10 ** (-5))):
#        print ("matrix is non invertible")
#    if (np.less(X_evalue_abs, (10 ** (-5)))):
#        print ("matrix is non invertible")
    
#    X_covariance = np.dot(X.T, X)
    X_covariance = np.matmul(X.T, X)
    X_evalue = np.linalg.eigvals(X_covariance)
    print (X_evalue)
    X_evalue_abs = np.absolute(X_evalue)
    X_smallest_evalue_abs_val = np.min(X_evalue_abs)
    print(X_smallest_evalue_abs_val)
    while X_smallest_evalue_abs_val < 0.00001:
        X_covariance += 0.1 * np.identity(X_covariance.shape[0])
        X_evalue = np.linalg.eigvals(X_covariance)
        X_evalue_abs = np.absolute(X_evalue)
        X_smallest_evalue_abs_val = np.min(X_evalue_abs)
        print(X_smallest_evalue_abs_val)
    print (X_evalue)   
    x2 = np.linalg.inv(X_covariance)
    x3 = np.matmul(x2, X.T)
    w = np.matmul(x3, y)
    return w


###### Q1.4 ######
def regularized_linear_regression(X, y, lambd):
    """
    Compute the weight parameter given X, y and lambda.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing feature.
    - y: A numpy array of shape (num_samples, ) containing label
    - lambd: a float number containing regularization strength
    Returns:
    - w: a numpy array of shape (D, )
    """
  #####################################################
  # TODO 4: Fill in your code here #
  #####################################################		
#    w = linear_regression_invertible(X, y)
    w = np.linalg.lstsq(np.dot(X.T, X) + lambd * np.identity(X.shape[1]), np.dot(X.T, y))[0]
    return w

###### Q1.5 ######
def tune_lambda(Xtrain, ytrain, Xval, yval):
    """
    Find the best lambda value.
    Inputs:
    - Xtrain: A numpy array of shape (num_training_samples, D) containing training feature.
    - ytrain: A numpy array of shape (num_training_samples, ) containing training label
    - Xval: A numpy array of shape (num_val_samples, D) containing validation feature.
    - yval: A numpy array of shape (num_val_samples, ) containing validation label
    Returns:
    - bestlambda: the best lambda you find in lambds
    """
    #####################################################
    # TODO 5: Fill in your code here #
    #####################################################
    best_err = 100
    for exp in range(-19, 20):		
        lambd = 10 ** exp
        w = regularized_linear_regression(Xtrain, ytrain, lambd)
        err = mean_absolute_error(w, Xval, yval)
        if (err < best_err):
            bestlambda = lambd
            best_err = err
    return bestlambda
    

###### Q1.6 ######
def mapping_data(X, power):
    """
    Mapping the data.
    Inputs:
    - X: A numpy array of shape (num_training_samples, D) containing training feature.
    - power: A integer that indicate the power in polynomial regression
    Returns:
    - X: mapped_X, You can manully calculate the size of X based on the power and original size of X
    """
    #####################################################
    # TODO 6: Fill in your code here #
    #####################################################
    new_X = np.array([[] for i in range(X.shape[0])])
    for p in range(1, power + 1):
       values = X ** p
       new_X = np.insert(new_X, [X.shape[1] * (p - 1)], values, axis=1)
    
    return new_X


