#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 19:15:04 2019

@author: vanish
"""

features = [[2,3], [3,1], [4,5]]
labels = [1,0,1]

points = [[3,1], [5,5], [1,3]]
from utils import Distances
k = KNN(2, Distances.euclidean_distance)
k.train(features, labels)
k.get_k_neighbors(points)
k.predict(points)


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
        print (features)
        norm_features = []
#        self.features = np.reshape(features, (3,2))
#        features = features / np.linalg.norm(features, ord=2, axis=1, keepdims=True)
#        norm_features = features / (np.sqrt(np.sum(np.dot(features, features))))
#        for x in range(len(features)):
        for x in range(len(features)):
#            denominator = np.sqrt(features[x] * features[x])
#            print (denominator)
            for y in range(len(features[x])):
#                denominator = features[x]*features[x]
                if features[x][y] == 0:
                    norm_features.append(0)
                else:
                    norm_features.append(features[x][y] / np.sqrt(np.sum(features[x] * features[x])))
        norm_features = np.reshape(norm_features, (features.shape[0],features.shape[1]))
        return norm_features.tolist()
        
        raise NotImplementedError





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
        
        self.distance_funcs = distance_funcs
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        dist_funcs_priority = {'Distances.euclidean_distance' : 1,
                               'Distances.minkowski_distance': 2,
                               'Distances.gaussian_kernel_distance': 3,
                               'Distances.inner_product_distance': 4,
                               'Distances.cosine_similarity_distance': 5,}
        best_f1 = 0
        for selected_distance_func in distance_funcs:
#            print (selected_distance_func)
            for k in range(1,31,2):
                model = KNN(k, distance_funcs[selected_distance_func])
                model.train(x_train, y_train)
                y_val_dash = model.predict(x_val)
                f1_score_value = f1_score(y_val, y_val_dash)
                print (selected_distance_func, "-", f1_score_value)
                if self.best_k == None:
                    self.best_k = k
                    self.best_distance_function = distance_funcs[selected_distance_func]
                    self.best_model = KNN(k, distance_funcs[selected_distance_func])
                    best_f1 = f1_score_value
                elif f1_score_value > best_f1:
                    self.best_k = k
                    self.best_distance_function = distance_funcs[selected_distance_func]
                    self.best_model = KNN(k, distance_funcs[selected_distance_func])
                    best_f1 = f1_score_value
                elif f1_score_value == best_f1:
                    current_best_dist_func = dist_funcs_priority[self.best_distance_function]
                    maybe_best_dist_func = dist_funcs_priority[distance_funcs[selected_distance_func]]
                    if (maybe_best_dist_func < current_best_dist_func):
                        self.best_k = k
                        self.best_distance_function = distance_funcs[selected_distance_func]
                        self.best_model = KNN(k, distance_funcs[selected_distance_func])
                    
        print("Best f1", best_f1, "Best Distance function", self.best_distance_function)
        return
        raise NotImplementedError