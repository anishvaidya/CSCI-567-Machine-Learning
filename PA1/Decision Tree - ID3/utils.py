import numpy as np
from copy import deepcopy

# TODO: Information Gain function
def Information_Gain(S, branches):
    # S: float
    # branches: List[List[int]] num_branches * num_cls
    # return: float
    branches = np.array(branches)
    new_branches = branches.copy()
    new_branches[new_branches == 0] = 1
#    for i in range (len(branches)):
#        for j in range (len(branches[i])):
#            entropy_of_system = branches[i]/
    
    
#    for i in range (len(branches[0])):
#        value = (np.sum(branches, axis = 0)[i] / np.sum(branches))
#        print (value)
#        entropy_of_system += -1 * (value * (np.log(value) / np.log(len(branches[0]))))
#        print (entropy_of_system)
    
    entropy_of_system = S
    entropy_of_attribute = 0    
    for i in range (len(branches)):
#        print (branches[i])
        if (np.sum(branches, axis = 1)[i] == 0):
            entropy_of_attribute += 0
        elif(np.sum(branches, axis = 1)[i] == 1):
            entropy_of_attribute += 0
        else:
            z = np.sum(branches, axis = 1)[i]
            if z == 0:
                z = 1
            log_value = (np.log2(new_branches[i]) - np.log2(z) )
            value = np.sum((branches[i] / np.sum(branches, axis = 1)[i]) * log_value)
            print (value)
            entropy_of_attribute += -1 * (value * (np.sum(branches, axis = 1)[i] / np.sum(branches)))
        print (entropy_of_attribute)
        
    information_gain = entropy_of_system - entropy_of_attribute
    return information_gain
    
#    for i in range (len(branches)):
##        print (branches[i])
#        if (np.sum(branches, axis = 1)[i] == 0):
#            entropy_of_attribute += 0
#        elif(np.sum(branches, axis = 1)[i] == 1):
#            entropy_of_attribute += 0
#        else:            
##            log_value = (np.log (new_branches[i] / np.sum(branches, axis = 1)[i]) / np.log(len(branches[0])))
#            log_value = (np.log2(new_branches[i] / np.sum(branches, axis = 1)[i]) )
#            value = np.sum((branches[i] / np.sum(branches, axis = 1)[i]) * log_value)
#            print (value)
#            entropy_of_attribute += -1 * (value * (np.sum(branches, axis = 1)[i] / np.sum(branches)))
#        print (entropy_of_attribute)
#        
#    information_gain = entropy_of_system - entropy_of_attribute
#    return information_gain
    raise NotImplementedError


# TODO: implement reduced error prunning function, pruning your tree on this function
def reduced_error_prunning(decisionTree, X_test, y_test):
    # decisionTree
    # X_test: List[List[any]]
    # y_test: List
    print ("pruning started")
#    print (decisionTree)
#    prediction_labels = decisionTree.predict(X_test)
#    n_total = len(y_test)
#    acc_count = 0
#    for i in range(len(y_test)):
#        if (prediction_labels[i] == y_test[i]):
#            acc_count += 1
#    test_accuracy = acc_count / n_total
#    print ("test accuracy is ", test_accuracy)
#    print (decisionTree.root_node.children)
#    
#    root = decisionTree.root_node
#    children = root.children
#    list = []
#    for child in children:
#        children.pop(child)
#        
#        print (child.children, " ", child.splittable)
#        
#        for i in range(len(child.children)):
#            reduced_error_prunning(child, X_test, y_test)
#    new_x_test = []
#    for feature in X_test:
#        new_x_test.append([value for value in feature])        
    prediction_labels = decisionTree.predict(X_test)
    n_total = len(y_test)
    acc_count = 0
    for i in range(len(y_test)):
        if (prediction_labels[i] == y_test[i]):
            acc_count += 1
    test_acc = acc_count / n_total
    print ("test accuracy is ", test_acc)
    old_acc = (None, test_acc)
    print ('old acc ', old_acc)
    counter = 0
#    track = []
    while (True):
        counter += 1
        print ('counter', counter)
        new_acc = old_acc
        print ('new acc', new_acc)
        frontier = [decisionTree.root_node]
        print ('frontier is ', frontier)
#        track.append(1)
        while len(frontier) != 0:
            node = frontier.pop(0)
            print ('selected node', node)
            if node.splittable:
                frontier.extend([child for child in node.children if child.splittable == True])
                print ('frontier is ', frontier)
                node.splittable = False #prune
                
#                new_x_test = []
#                for feature in X_test:
#                    new_x_test.append([value for value in feature])
                
                prediction_labels = decisionTree.predict(X_test)
                n_total = len(y_test)
                acc_count = 0
                for i in range(len(y_test)):
                    if (prediction_labels[i] == y_test[i]):
                        acc_count += 1
                test_acc = acc_count / n_total
                print ("test accuracy is ", test_acc)
#                print(track)
                if test_acc >= new_acc[1]:
                    new_acc = (node, test_acc)
                node.splittable = True
        if new_acc == old_acc:
            print ('new acc == old acc', new_acc, ' breaking')
            break
        elif new_acc[1] >= old_acc[1]:
            new_acc[0].splittable = False
            old_acc = new_acc
            
    return 
    raise NotImplementedError


# print current tree
def print_tree(decisionTree, node=None, name='branch 0', indent='', deep=0):
    if node is None:
        node = decisionTree.root_node
    print(name + '{')

    print(indent + '\tdeep: ' + str(deep))
    string = ''
    label_uniq = np.unique(node.labels).tolist()
    for label in label_uniq:
        string += str(node.labels.count(label)) + ' : '
    print(indent + '\tnum of samples for each class: ' + string[:-2])

    if node.splittable:
#        print(indent + '\tsplit by dim {:d}'.format(node.dim_split))
        for idx_child, child in enumerate(node.children):
            print_tree(decisionTree, node=child, name='\t' + name + '->' + str(idx_child), indent=indent + '\t', deep=deep+1)
    else:
        print(indent + '\tclass:', node.cls_max)
    print(indent + '}')

