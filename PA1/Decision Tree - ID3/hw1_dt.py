import numpy as np
import utils as Util


class DecisionTree():
    def __init__(self):
        self.clf_name = "DecisionTree"
        self.root_node = None

    def train(self, features, labels):
        # features: List[List[float]], labels: List[int]
        # init
        assert (len(features) > 0)
        num_cls = np.unique(labels).size

        # build the tree
        self.root_node = TreeNode(features, labels, num_cls)
        if self.root_node.splittable:
            self.root_node.split()

        return

    def predict(self, features):
        # features: List[List[any]]
        # return List[int]
        y_pred = []
        for idx, feature in enumerate(features):
            pred = self.root_node.predict(feature)
            y_pred.append(pred)
        return y_pred
    
    def test_acc(self, X_test, y_test):
        prediction_labels = self.predict(X_test)
        n_total = len(y_test)
        acc_count = 0
        for i in range(len(y_test)):
            if (prediction_labels[i] == y_test[i]):
                acc_count += 1
        test_accuracy = acc_count / n_total
        print ("test accuracy is ", test_accuracy)
#        print (self.root_node.children)
        return test_accuracy


class TreeNode(object):
    def __init__(self, features, labels, num_cls):
        # features: List[List[any]], labels: List[int], num_cls: int
        self.features = features
        self.labels = labels
        self.children = []
        self.num_cls = num_cls
        # find the most common labels in current node
        count_max = 0
        for label in np.unique(labels):
            if self.labels.count(label) > count_max:
                count_max = labels.count(label)
                self.cls_max = label
                # splitable is false when all features belongs to one class
        if len(np.unique(labels)) < 2:
            self.splittable = False
        else:
            self.splittable = True

        self.dim_split = None  # the index of the feature to be split

        self.feature_uniq_split = None  # the possible unique values of the feature to be split

    #TODO: try to split current node
    def split(self):
        self.feature_uniq_split = []
        S = 0
#        self.features = [[0, 0], [1, 0], [0, 1], [1, 1], [0, 0], [1, 0], [0, 1], [1, 1]]
#        self.labels = [1,2,0,1,2,0,0,0]
#        self.features = [['a', 'b'], ['b', 'a'], ['b', 'c'], ['a', 'c']]
#        self.labels = [0, 0, 1, 1]
        np_features = np.array(self.features)
        features_T = np.transpose(np_features)
        np_labels = np.array(self.labels)
        print ("features transpose", features_T)
        print ("labels", np_labels)

        unique_labels = list(np.unique(np_labels))

        for label in np.unique(self.labels):
            p_of_label = self.labels.count(label) / len(self.labels)            
#            print (p_of_label)
            S -= p_of_label * np.log2(p_of_label)
        print (S)

        #unique_labels = list(np.unique(np_labels))
#        features_transpose = np.transpose(features)
        list_info_gain = [] 
        for i in range(len(features_T)):
            attribute_value = list(np.unique(features_T[i]))
            current_attribute = features_T[i]
            branches = []
            for idx in range(len(attribute_value)):
                count_of_labels = []
                for j in range(self.num_cls):
                    counter = 0
                    for attr_val,k in zip(current_attribute,np_labels):
                        if attr_val == attribute_value[idx] and k == unique_labels[j]:
                            counter += 1
                    count_of_labels.append(counter)
                branches.append(count_of_labels)
            print (branches)
            list_info_gain.append(Util.Information_Gain(S, branches))
#        for i in range(len(unique_labels)):
#            list_info_gain.append(Util.Information_Gain(S, branches))
        print(list_info_gain)
        if (list_info_gain == []):
            self.dim_split = None
            self.feature_uniq_split = None
            self.splittable = False
            return
        
        if (max(list_info_gain) == 0):
            self.dim_split = None
            self.feature_uniq_split = None
            self.splittable = False
            return
        
        selected_attr_to_split = list_info_gain.index(max(list_info_gain))
        
        list_max_info_gain = []
        count_attr_clash = []
        
        for index,info_gain_value in enumerate(list_info_gain):
            if (info_gain_value == max(list_info_gain)):
                list_max_info_gain.append(index)
        
        for index in list_max_info_gain:
            count_attr_clash.append(np.unique(features_T[index]).size)
          
        selected_attr_clash = count_attr_clash.index(max(count_attr_clash))
        selected_attr_to_split = list_max_info_gain[selected_attr_clash]
            
        
        print (selected_attr_to_split)
        self.dim_split = selected_attr_to_split
#        self.feature_uniq_split.append(np.unique(features_T[self.dim_split]))
        self.feature_uniq_split = list(np.unique(features_T[self.dim_split]))
#        self.feature_uniq_split.sort()
#        self.feature_uniq_split = np.array(self.feature_uniq_split)
#        print(self.feature_uniq_split.dtype)
        print ("dim split", self.dim_split)        
        print ("feature unique split", self.feature_uniq_split)
        for selected_attr_value in self.feature_uniq_split:
            new_features = []
            new_labels = []
            print (selected_attr_value)
#            new_features = np.delete(self.features, self.dim_split, axis = 1)
            #check this loop
            print (np_features.shape[0])
            for i in range(np_features.shape[0]):
                    if (np_features[i][self.dim_split] == selected_attr_value):
                        new_features.append(self.features[i][0:self.dim_split] + self.features[i][self.dim_split+1:])
#                        new_features.append(str(np_features[i, 0:self.dim_split]) + str(np_features[i, self.dim_split+1:]))
#                        new_features.append(self.features[i][:]
                        new_labels.append(self.labels[i])
#                    print ("new features before transformation ", new_features)
                    new_features = [x for x in new_features if x]
            
            new_num_classes = np.unique(new_labels).size
            print (new_features, " size is ", len(new_features))
            print (new_features , " these are new features")
            print (new_labels, " size is", len(new_labels))
            print (new_labels, " these are new labels")
            self.children.append(TreeNode(new_features, new_labels, new_num_classes))
        
        
        
        for child in self.children:
            if (child.splittable):
                child.split()
#        return new_features
#        raise NotImplementedError

    # TODO: predict the branch or the class
    def predict(self, feature):
        # feature: List[any]
        # return: int
        
        if (self.splittable):
            new_x_test = []
            for x in feature:
                new_x_test.append(x)
#            print("new x test ", new_x_test)
#            print ("feature ", feature)
#            current_selected_attr = feature[self.dim_split]
#            selected_branch = self.feature_uniq_split.index(current_selected_attr)
##            print ("selected branch", selected_branch)
#            
#            feature.pop(self.dim_split)
#            new_branch = self.children[selected_branch]
##            print("new branch ", new_branch)
##            del feature[self.dim_split]
#            return (new_branch.predict(feature))
            current_selected_attr = new_x_test[self.dim_split]
#            print ("current selected attribute", current_selected_attr)
            selected_branch = self.feature_uniq_split.index(current_selected_attr)
#            print ("selected branch", selected_branch)
            
            new_x_test.pop(self.dim_split)
            new_branch = self.children[selected_branch]
#            print("new branch ", new_branch)
#            del feature[self.dim_split]
            return (new_branch.predict(new_x_test))
        else:
            return self.cls_max
        
        raise NotImplementedError
        

        

    
#    def val_accuracy(self, prediction_labels, test_labels):
#        n_total = len(test_labels)
#        acc_count = 0
#        for i in range(len(test_labels)):
#            if (prediction_labels[i] == test_labels[i]):
#                acc_count += 1
#        validation_accuracy = acc_count / n_total
#        return validation_accuracy
