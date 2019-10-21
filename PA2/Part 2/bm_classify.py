import numpy as np


def binary_train(X, y, loss="perceptron", w0=None, b0=None, step_size=0.5, max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - y: binary training labels, a N dimensional numpy array where 
    N is the number of training points, indicating the labels of 
    training data
    - loss: loss type, either perceptron or logistic
    - step_size: step size (learning rate)
	- max_iterations: number of iterations to perform gradient descent

    Returns:
    - w: D-dimensional vector, a numpy array which is the weight 
    vector of logistic or perceptron regression
    - b: scalar, which is the bias of logistic or perceptron regression
    """
    N, D = X.shape
    assert len(np.unique(y)) == 2


    w = np.zeros(D)
    if w0 is not None:
        w = w0
    
    b = 0
    if b0 is not None:
        b = b0

    if loss == "perceptron":
        ############################################
        # TODO 1 : Edit this if part               #
        #          Compute w and b here            #
#        y[y == 0] = -1
        y = np.where(y == 0, -1, 1)
#        w = np.zeros(D)
#        b = 0
#        for iteration in range (0, max_iterations):
#            z = y * (np.dot(X, w) + b)
##            print("z", z.shape)
#            if (np.where(z<=0)):
#                w += (step_size * np.dot(X.T, y) / N)
#                b += (step_size * y / N)
#            b = b[0,]
        print(w.shape)   
        for iteration in range(0, 2):
            z = y * (np.dot(X, w) + b)
            z1 = np.where(z <= 0, 1, 0)
            # z1 * y * X     (D,)
            # n,   n,  n,d
            print('z1', z1.shape)
            a = np.multiply(z1, y)
            print('a', a.shape)
            c = np.dot(a, X)
#            c = np.sum(c.T)
            print('c', c.shape)
            w += step_size * c / N
            b += np.sum(step_size * z1 * y / N)
            
            
#            for i in range(0, N):
#                if (z[i] <= 0):
#                    w += step_size * y[i] * X[i] / N
#                    b += step_size * y[i] / N
                    
            
#            print("w", w.shape)
#            print("b", b.shape)
        ############################################
        

    elif loss == "logistic":
        ############################################
        # TODO 2 : Edit this if part               #
        #          Compute w and b here            #
#        y[y == 0] = -1
        y = np.where(y == 0, -1, 1)
        for iteration in range (0, max_iterations):
            z = y * (np.dot(X,w) + b)
            
            w_gradient = np.dot(X.T, (y * sigmoid(-z)))
            b_gradient = np.sum(y * sigmoid(-z))
            
            w += step_size * w_gradient / N
            b += step_size * b_gradient / N
#            print(w.shape)
#            print(b)
#            print ("b",b.shape)
        ############################################
        

    else:
        raise "Loss Function is undefined."

    assert w.shape == (D,)
    return w, b

def sigmoid(z):
    
    """
    Inputs:
    - z: a numpy array or a float number
    
    Returns:
    - value: a numpy array or a float number after computing sigmoid function value = 1/(1+exp(-z)).
    """

    ############################################
    # TODO 3 : Edit this part to               #
    #          Compute value                   #
    value = 1 / (1 + np.exp(-z))
#    print(value)
    ############################################
    
    return value

def binary_predict(X, w, b, loss="perceptron"):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - w: D-dimensional vector, a numpy array which is the weight 
    vector of your learned model
    - b: scalar, which is the bias of your model
    - loss: loss type, either perceptron or logistic
    
    Returns:
    - preds: N dimensional vector of binary predictions: {0, 1}
    """
    N, D = X.shape
    
    if loss == "perceptron":
        ############################################
        # TODO 4 : Edit this if part               #
        #          Compute preds                   #
        preds = np.zeros(N)
        z = np.dot(X, w) + b
        preds = np.sign(z)
        preds[preds == -1] = 0
        print (preds)
        
        ############################################
        

    elif loss == "logistic":
        ############################################
        # TODO 5 : Edit this if part               #
        #          Compute preds                   #
        preds = np.zeros(N)
        z = np.dot(X, w.T) + b
        preds = sigmoid(z)
#        preds = np.sign(z)
#        print("preds", preds.shape)
#        preds = np.where(preds>=0, preds, 1)
#        preds = np.where(preds<0, preds, 0)
        preds[preds > 0.5] = 1
        preds[preds <= 0.5] = 0
#        print(preds)
#        for i in preds:
#            if i >= 0.5: i = 1
#        else: i = 0
#        preds[np.where(preds >= 0.5)] = 1
#        preds[np.where(preds < 0.5)] = 0]
        ############################################
        

    else:
        raise "Loss Function is undefined."
    

    assert preds.shape == (N,) 
    return preds



def multiclass_train(X, y, C,
                     w0=None, 
                     b0=None,
                     gd_type="sgd",
                     step_size=0.5, 
                     max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - y: multiclass training labels, a N dimensional numpy array where
    N is the number of training points, indicating the labels of 
    training data
    - C: number of classes in the data
    - gd_type: gradient descent type, either GD or SGD
    - step_size: step size (learning rate)
    - max_iterations: number of iterations to perform gradient descent

    Returns:
    - w: C-by-D weight matrix of multinomial logistic regression, where 
    C is the number of classes and D is the dimensionality of features.
    - b: bias vector of length C, where C is the number of classes
    """

    N, D = X.shape

    w = np.zeros((C, D))
    if w0 is not None:
        w = w0
    
    b = np.zeros(C)
    if b0 is not None:
        b = b0

    np.random.seed(42)
    if gd_type == "sgd":
        ############################################
        # TODO 6 : Edit this if part               #
        #          Compute w and b                 #
        xb = np.ones((N,1))
        new_X = np.append(X, xb, axis = 1)
        
        b = b.reshape((C,1))
        w = np.append(w, b, axis = 1)
        
        for iteration in range(0, max_iterations):
            
            n = np.random.choice(N)
            selected_X = new_X[n]
#            print()
            selected_X = np.reshape(selected_X, (D + 1, 1))
#            print(selected_X.shape)
            one_hot_labels = np.zeros((1, C))
            one_hot_labels[0][y[n]] = 1
#            print('o_h_labels', one_hot_labels.shape)
            z = np.dot(w, selected_X)
            
            numerator = np.exp(z - np.amax(z))
            denominator = np.sum(numerator, axis = 0)
            
            softmax = (numerator / denominator) - one_hot_labels.T
            update = np.dot(softmax, selected_X.T)
            
            w = w - step_size * update
        b = w[:, -1]
        w = w[:, :-1]
        ############################################
        

    elif gd_type == "gd":
        ############################################
        # TODO 7 : Edit this if part               #
        #          Compute w and b                 #
#        w = np.zeros((C, D))
#        b = np.zeros(C)
        xb = np.ones((N,1))
        new_X = np.append(X, xb, axis = 1)
        
        b = b.reshape((C,1))
        w = np.append(w, b, axis = 1)
        
        one_hot_labels = np.zeros((N, C))
        for x, val in enumerate(y):
            one_hot_labels[x][val] = 1
        one_hot_labels_new = np.where(y == 1, 1, 0)
        print(one_hot_labels.shape)
        print(one_hot_labels_new.shape)
        
        
        for iteration in range(0, max_iterations):
            
            z = np.dot(w, new_X.T)
            
            numerator = np.exp(z - np.amax(z))
            denominator = np.sum(numerator, axis = 0)
            
            softmax = (numerator / denominator) - one_hot_labels.T
            update = np.dot(softmax, new_X)
            
            w = w - step_size * update / N
        b = w[:, -1]
        w = w[:, :-1]
        ############################################
        

    else:
        raise "Type of Gradient Descent is undefined."
    

    assert w.shape == (C, D)
    assert b.shape == (C,)

    return w, b


def multiclass_predict(X, w, b):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - w: weights of the trained multinomial classifier, C-by-D 
    - b: bias terms of the trained multinomial classifier, length of C
    
    Returns:
    - preds: N dimensional vector of multiclass predictions.
    Outputted predictions should be from {0, C - 1}, where
    C is the number of classes
    """
    N, D = X.shape
    ############################################
    # TODO 8 : Edit this part to               #
    #          Compute preds                   #
#    preds = np.zeros(N)
    xb = np.ones((N,1))
    C = w.shape[0]
    new_X = np.append(X, xb, axis = 1)
    b = b.reshape((C,1))
    w = np.append(w, b, axis = 1)
    z = np.dot(w, new_X.T)
    numerator = np.exp(z - np.amax(z))
    denominator = np.sum(numerator, axis = 0)
    softmax = (numerator) / (denominator)
    preds = np.argmax(softmax, axis = 0)
    ############################################

    assert preds.shape == (N,)
    return preds




        