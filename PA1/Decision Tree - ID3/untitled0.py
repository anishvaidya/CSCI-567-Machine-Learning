#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 18:59:00 2019

@author: vanish
"""
import numpy as np
features = [['a', 'b'], ['b', 'a'], ['b', 'c'], ['c', 'b']]
labels = [0, 0, 1, 1]
num_cls = 2
features = np.array(features)
labels = np.array(labels)
unique_labels = list(np.unique(labels))
features_transpose = np.transpose(features)
for i in range(len(features_transpose)):
    unique_val = list(np.unique(features_transpose[i]))
    attribute = features_transpose[i]
    branches = []
    for idx in range(len(unique_val)):
        ans = []
        for s in range(num_cls):
            k = 0
            for val,x in zip(attribute,labels):
                if val == unique_val[idx] and x == unique_labels[s]:
                    k += 1
            ans.append(k)
        branches.append(ans)
print (branches)



# check this code
        for feature in range(len(features_T)):
            attribute_value = list(np.unique(features_T[feature]))
            current_attribute = features_T[feature]
            branches = []
            for i in range(len(attribute_value)):
                count_of_labels = []
                for j in range(np.unique(self.labels).size):
                    counter = 0
                    for att_value, k in zip(current_attribute, np_labels):
                        if (att_value == current_attribute[i] and k == unique_labels[j]):
                            counter += 1
                    count_of_labels.append(counter)
                branches.append(count_of_labels)
        print (branches)
        
        
        
for selected_attr_value in self.feature_uniq_split:
            new_features = []
            new_labels = []
            print (selected_attr_value)
#            new_features = np.delete(self.features, self.dim_split, axis = 1)
            #check this loop
            for i in range(np_features.shape[0]):
                    if (np_features[i][self.dim_split] == selected_attr_value):
#                        new_features = np.delete(self.features, self.dim_split, axis = 1)
#                        new_features = np.delete(new_features, )
#                        new_features.append(self.features[i][0:self.dim_split, self.dim_split+1:])
                        # this worked
                        new_features.append(self.features[i][0:self.dim_split])
                        new_features.append(self.features[i][self.dim_split+1:])
                        # till here
#                        new_features = np.append(new_features,self.features[i][:])
#                        new_features = np.delete(np.array(new_features), self.dim_split, axis = 1)
#                        new_features = np.delete(np.array(new_features), self.dim_split, axis = 1)
                        
                        new_labels.append(self.labels[i])
            new_features = [x for x in new_features if x]