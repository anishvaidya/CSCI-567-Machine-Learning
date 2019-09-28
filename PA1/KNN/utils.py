import numpy as np
from knn import KNN

############################################################################
# DO NOT MODIFY ABOVE CODES
############################################################################


# TODO: implement F1 score
def f1_score(real_labels, predicted_labels):
    """
    Information on F1 score - https://en.wikipedia.org/wiki/F1_score
    :param real_labels: List[int]
    :param predicted_labels: List[int]
    :return: float
    """
    assert len(real_labels) == len(predicted_labels)
    true_p = 0
    true_n = 0
    false_p = 0
    false_n = 0
    for x in range(len(real_labels)):
        if real_labels[x] == 1 and predicted_labels[x] ==1 :
            true_p += 1 
        elif real_labels[x] == 0 and predicted_labels[x] ==0 :
            true_n += 1
        elif real_labels[x] == 0 and predicted_labels[x] ==1 :
            false_p += 1
        elif real_labels[x] == 1 and predicted_labels[x] ==0 :
            false_n += 1   
#    print (true_p, "-", true_n, "-", false_p, "-", false_n)
    if (true_p == 0  and true_n == 0 and false_p == 0):
        f1_score = 1
    elif (true_p == 0 and (true_n > 0 or false_n > 0)):
        f1_score = 0
    else:    
        precision = true_p / (true_p + false_p)
        recall = true_p / (true_p + false_n)
#    f1_score = 2 / ( ((true_p + false_p) / true_p) + ((true_p + false_n) / true_p))
        f1_score = 2 * (1 / ((1 / precision) + (1 / recall)))
    return f1_score
    raise NotImplementedError


class Distances:
    @staticmethod
    # TODO
    def minkowski_distance(point1, point2):
        """
        Minkowski distance is the generalized version of Euclidean Distance
        It is also know as L-p norm (where p>=1) that you have studied in class
        For our assignment we need to take p=3
        Information on Minkowski distance - https://en.wikipedia.org/wiki/Minkowski_distance
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
#        p = 3
#        a = np.subtract(point1, point2)
##        print (a)
#        b = np.absolute(a)
##        print (b)
#        c = b ** p
##        print (c)
#        d = np.sum(c)
##        print (d)
#        e = d ** (1/p)
##        print (e)
#        return e
        p = 3
        #return np.sum(np.absolute(point1 - point2) ** p) ** 1/p
        #return (np.linalg.norm(np.subtract(point1, point2)) ** p) ** 1/p
        #return np.absolute((np.sum(point1 - point2) ** p) ** 1/p)
        point1 = np.array(point1)
        point2 = np.array(point2)
        return np.sum(np.absolute(np.subtract(point1, point2)) ** p) ** (1/p)
        raise NotImplementedError
#        return float(np.sum(np.absolute(np.subtract(point1, point2)) ** p) ** 1//p)
#        return (np.linalg.norm(np.subtract(point1, point2)) ** p) ** 1/p
#        return np.absolute((np.sum(point1 - point2) ** p) ** 1/p)
        raise NotImplementedError

    @staticmethod
    # TODO
    def euclidean_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
#        dist = np.sqrt(np.square(point1 - point2), dtype = 'float64')
#        return dist
        point1 = np.array(point1)
        point2 = np.array(point2)
        return np.sqrt(np.sum((point1 - point2) ** 2))
        raise NotImplementedError

    @staticmethod
    # TODO
    def inner_product_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        point1 = np.array(point1)
        point2 = np.array(point2)
        return (np.inner(point1, point2))
        raise NotImplementedError

    @staticmethod
    # TODO
    def cosine_similarity_distance(point1, point2):
        """
       :param point1: List[float]
       :param point2: List[float]
       :return: float
       """
        point1 = np.array(point1)
        point2 = np.array(point2)
        #return (np.dot(point1, point2) / (np.linalg.norm(point1) * np.linalg.norm(point2)))
        return (1 - (np.dot(point1, point2)/(np.sqrt(np.sum((point1) ** 2)) * np.sqrt(np.sum((point2) ** 2)))))
        raise NotImplementedError

    @staticmethod
    # TODO
    def gaussian_kernel_distance(point1, point2):
        """
       :param point1: List[float]
       :param point2: List[float]
       :return: float
       """
#        return np.exp(-1 // 2 * np.sum(point1 - point2) ** 2)
        point1 = np.array(point1)
        point2 = np.array(point2)
        return (- np.exp(-0.5 * (np.sum(np.multiply(point1 - point2, point1 - point2)))))
        raise NotImplementedError


class HyperparameterTuner:
    def __init__(self):
        self.best_k = None
        self.best_distance_function = None
        self.best_scaler = None
        self.best_model = None

    # TODO: find parameters with the best f1 score on validation dataset
    def tuning_without_scaling(self, distance_funcs, x_train, y_train, x_val, y_val):
        """
        In this part, you should try different distance function you implemented in part 1.1, and find the best k.
        Use k range from 1 to 30 and increment by 2. Use f1-score to compare different models.

        :param distance_funcs: dictionary of distance functions you must use to calculate the distance.
            Make sure you loop over all distance functions for each data point and each k value.
            You can refer to test.py file to see the format in which these functions will be
            passed by the grading script
        :param x_train: List[List[int]] training data set to train your KNN model
        :param y_train: List[int] train labels to train your KNN model
        :param x_val:  List[List[int]] Validation data set will be used on your KNN predict function to produce
            predicted labels and tune k and distance function.
        :param y_val: List[int] validation labels

        Find(tune) best k, distance_function and model (an instance of KNN) and assign to self.best_k,
        self.best_distance_function and self.best_model respectively.
        NOTE: self.best_scaler will be None

        NOTE: When there is a tie, choose model based on the following priorities:
        Then check distance function  [euclidean > minkowski > gaussian > inner_prod > cosine_dist]
        If they have same distance fuction, choose model which has a less k.
        """
        
        # You need to assign the final values to these variables
        self.best_k = None
        self.best_distance_function = None
        self.best_model = None
#        
#        dist_funcs_priority = {'euclidean' : 1,
#                               'minkowski': 2,
#                               'gaussian': 3,
#                               'inner_prod': 4,
#                               'cosine_dist': 5,}
        best_f1 = 0
        for selected_distance_func in distance_funcs:
            for k in range(1, min(31, len(x_train) + 1), 2):
                model = KNN(k, distance_funcs[selected_distance_func])
                model.train(x_train, y_train)
                y_val_dash = model.predict(x_val)
                f1_score_value = f1_score(y_val, y_val_dash)
                print (selected_distance_func, "-", f1_score_value)
                if self.best_k == None:
                    self.best_k = k
#                    self.best_distance_function = distance_funcs[selected_distance_func]
                    self.best_distance_function = selected_distance_func
                    self.best_model = model
                    best_f1 = f1_score_value
                elif f1_score_value > best_f1:
                        self.best_k = k
                        self.best_distance_function = selected_distance_func
                        self.best_model = model
                        best_f1 = f1_score_value
#                elif f1_score_value == best_f1:
#                    current_best_dist_func = dist_funcs_priority[self.best_distance_function]
##                    maybe_best_dist_func = dist_funcs_priority[distance_funcs[selected_distance_func]]
#                    maybe_best_dist_func = dist_funcs_priority[selected_distance_func]
#                    if (maybe_best_dist_func < current_best_dist_func):
#                        self.best_k = k
##                        self.best_distance_function = distance_funcs[selected_distance_func]
#                        self.best_distance_function = selected_distance_func
#                        self.best_model = KNN(k, distance_funcs[selected_distance_func])
##                        print("zaala bc")
                    
        print("Best f1", best_f1, "Best Distance function", self.best_distance_function)
        return
        raise NotImplementedError

#    # TODO: find parameters with the best f1 score on validation dataset, with normalized data
    def tuning_with_scaling(self, distance_funcs, scaling_classes, x_train, y_train, x_val, y_val):
        """
        This part is similar to Part 1.3 except that before passing your training and validation data to KNN model to
        tune k and disrance function, you need to create the normalized data using these two scalers to transform your
        data, both training and validation. Again, we will use f1-score to compare different models.
        Here we have 3 hyperparameters i.e. k, distance_function and scaler.

        :param distance_funcs: dictionary of distance funtions you use to calculate the distance. Make sure you
            loop over all distance function for each data point and each k value.
            You can refer to test.py file to see the format in which these functions will be
            passed by the grading script
        :param scaling_classes: dictionary of scalers you will use to normalized your data.
        Refer to test.py file to check the format.
        :param x_train: List[List[int]] training data set to train your KNN model
        :param y_train: List[int] train labels to train your KNN model
        :param x_val: List[List[int]] validation data set you will use on your KNN predict function to produce predicted
            labels and tune your k, distance function and scaler.
        :param y_val: List[int] validation labels

        Find(tune) best k, distance_funtion, scaler and model (an instance of KNN) and assign to self.best_k,
        self.best_distance_function, self.best_scaler and self.best_model respectively

        NOTE: When there is a tie, choose model based on the following priorities:
        For normalization, [min_max_scale > normalize];
        Then check distance function  [euclidean > minkowski > gaussian > inner_prod > cosine_dist]
        If they have same distance function, choose model which has a less k.
        """
        
        # You need to assign the final values to these variables
        self.best_k = None
        self.best_distance_function = None
        self.best_scaler = None
        self.best_model = None
        
        self.scaling_classes_priority = [
                        'min_max_scale',
                        'normalize']
        
        self.dist_funcs_priority = ['euclidean',
                               'minkowski',
                               'gaussian',
                               'inner_prod',
                               'cosine_dist']
        best_f1 = 0
        for scaling_class in self.scaling_classes_priority:
            scaler = scaling_classes[scaling_class]()
            x_train_scaled = scaler.__call__(x_train)
            x_val_scaled = scaler.__call__(x_val)
            for selected_distance_func in self.dist_funcs_priority:
                for k in range(1, min(31, len(x_train) + 1), 2):
                    model = KNN(k, distance_funcs[selected_distance_func])
                    model.train(x_train_scaled, y_train)
                    y_val_dash = model.predict(x_val_scaled)
                    f1_score_value = f1_score(y_val, y_val_dash)
                    print (selected_distance_func, "-", f1_score_value)
                    if self.best_k == None:
                        self.best_k = k
    #                    self.best_distance_function = distance_funcs[selected_distance_func]
                        self.best_distance_function = selected_distance_func
                        self.best_scaler = scaling_class
                        self.best_model = model
                        best_f1 = f1_score_value
                    elif f1_score_value > best_f1:
                            self.best_k = k
                            self.best_distance_function = selected_distance_func
                            self.best_scaler = scaling_class
                            self.best_model = model
                            best_f1 = f1_score_value
#                    elif f1_score_value == best_f1:
#                        if (self.scaling_classes_priority[scaling_class] < self.scaling_classes_priority[self.best_scaler]):
#                            self.best_k = k
#                            self.best_distance_function = selected_distance_func
#                            self.best_scaler = scaling_class
#                            self.best_model = model
#                            best_f1 = f1_score_value
#                        elif (self.dist_funcs_priority[selected_distance_func] < self.dist_funcs_priority[self.best_distance_function]):
#                            self.best_k = k
#    #                        self.best_distance_function = distance_funcs[selected_distance_func]
#                            self.best_distance_function = selected_distance_func
#                            self.best_scaler = scaling_class
#                            self.best_model = model
#                            best_f1 = f1_score_value
#                                
#                        elif k < self.best_k:
#                            self.best_k = k
#                            self.best_distance_function = selected_distance_func
#                            self.best_scaler = scaling_class
#                            self.best_model = model 
#                            best_f1 = f1_score_value
                    
                    
#                    elif f1_score_value == best_f1:
#                        current_best_dist_func = dist_funcs_priority[self.best_distance_function]
#    #                    maybe_best_dist_func = dist_funcs_priority[distance_funcs[selected_distance_func]]
#                        maybe_best_dist_func = dist_funcs_priority[selected_distance_func]
#                        if (maybe_best_dist_func < current_best_dist_func):
#                            self.best_k = k
#    #                        self.best_distance_function = distance_funcs[selected_distance_func]
#                            self.best_distance_function = selected_distance_func
#                            self.best_model = KNN(k, distance_funcs[selected_distance_func])
        print ("best f1 score - ", best_f1)
        return        
        raise NotImplementedError


class NormalizationScaler:
    def __init__(self):
        pass

    # TODO: normalize data
    def __call__(self, features):
        """
        Normalize features for every sample

        Example
        features = [[3, 4], [1, -1], [0, 0]]
        return [[0.6, 0.8], [0.707107, -0.707107], [0, 0]]

        :param features: List[List[float]]
        :return: List[List[float]]
        """
        self.features = np.array(features)
#        self.features = np.array(features)
#        print (self.features)
        norm_features = []
#        self.features = np.reshape(features, (3,2))
#        features = features / np.linalg.norm(features, ord=2, axis=1, keepdims=True)
#        norm_features = features / (np.sqrt(np.sum(np.dot(features, features))))
#        for x in range(len(features)):
        for i in range(len(self.features)):
#            denominator = np.sqrt(features[x] * features[x])
#            print (denominator)
            for j in range(len(self.features[i])):
#                denominator = features[x]*features[x]
                if self.features[i][j] == 0:
                    norm_features.append(0)
                else:
                    norm_features.append(self.features[i][j] / np.sqrt(np.sum(self.features[i] * self.features[i])))
        norm_features = np.reshape(norm_features, (self.features.shape[0], self.features.shape[1]))
        return norm_features.tolist()
        raise NotImplementedError


class MinMaxScaler:
    """
    Please follow this link to know more about min max scaling
    https://en.wikipedia.org/wiki/Feature_scaling
    You should keep some states inside the object.
    You can assume that the parameter of the first __call__
    will be the training set.

    Hints:
        1. Use a variable to check for first __call__ and only compute
            and store min/max in that case.

    Note:
        1. You may assume the parameters are valid when __call__
            is being called the first time (you can find min and max).

    Example:
        train_features = [[0, 10], [2, 0]]
        test_features = [[20, 1]]

        scaler1 = MinMaxScale()
        train_features_scaled = scaler1(train_features)
        # train_features_scaled should be equal to [[0, 1], [1, 0]]

        test_features_scaled = scaler1(test_features)
        # test_features_scaled should be equal to [[10, 0.1]]

        new_scaler = MinMaxScale() # creating a new scaler
        _ = new_scaler([[1, 1], [0, 0]]) # new trainfeatures
        test_features_scaled = new_scaler(test_features)
        # now test_features_scaled should be [[20, 1]]

    """

    def __init__(self):
        self.first_call = True
        self.col_max = 0
        self.col_min = 0
        
        pass

    def __call__(self, features):
        """
        normalize the feature vector for each sample . For example,
        if the input features = [[2, -1], [-1, 5], [0, 0]],
        the output should be [[1, 0], [0, 1], [0.333333, 0.16667]]

        :param features: List[List[float]]
        :return: List[List[float]]
        """
        
        self.features = np.array(features)
#        self.features = np.array(features)
#        print (self.features)
        min_max_features = []
        if (self.first_call == True):
            self.col_min = np.amin(self.features, axis = 0)
            self.col_max = np.amax(self.features, axis = 0)
            self.first_call = False
        for i in range(len(self.features)):
            for j in range(len(self.features[i])):
#                min_max_features.append(self.features[i][j] / np.amax(self.features[i]))
                if (self.col_max[j] - self.col_min[j]) == 0:
                    min_max_features.append(1)
                else:
                    min_max_features.append((self.features[i][j] - self.col_min[j]) / (self.col_max[j]- self.col_min[j]))
#                    print(min_max_features)
                
        min_max_features = np.reshape(min_max_features, (self.features.shape[0], self.features.shape[1]))
        return min_max_features.tolist()
        
        raise NotImplementedError
