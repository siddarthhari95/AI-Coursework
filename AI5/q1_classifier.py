import pickle as pkl
import argparse
import csv
import numpy as np
import pandas as pd
from scipy.stats import chi2
import sys
import timeit

sys.setrecursionlimit(4000)

'''
TreeNode represents a node in your decision tree
TreeNode can be:
	- A non-leaf node:
		- data: contains the feature number this node is using to split the data
		- children[0]-children[4]: Each correspond to one of the values that the feature can take

	- A leaf node:
		- data: 'T' or 'F'
		- children[0]-children[4]: Doesn't matter, you can leave them the same or cast to None.
Taken the idea from Naveen Gad "https://github.com/naveengad/Clickstream-Mining"
'''
nodes_internal,node_leaf = 0,0


# DO NOT CHANGE THIS CLASS
class TreeNode():
    def __init__(self, data='T', children=[-1] * 5):
        self.nodes = list(children)
        self.data = data

    def save_tree(self, filename):
        obj = open(filename, 'w')
        pkl.dump(self, obj)


# Function which will calculate the p-value and says to stop expanding or not
def chisquare_split_stop(data_training, pred_attr, next_best, threshold_value):

    # counting the values by class
    class_count = data_training[pred_attr].value_counts()

    # calcuation of 'p' and 'n', positive and negative examples

    positive,negative,pos,neg,ct = 0.0,0.0,1,0,class_count.index.values

    # ct = class_count.index.values

    # for positive class
    if pos in ct:
        positive = float(class_count[1])

    # for negative class
    if neg in ct:
        negative = float(class_count[0])

    # adding 'positive' and 'negative' to get 'total', total number of examples
    total = positive + negative
    # start of calculation of statistic of interest
    S = 0.0

    for each in data_training[next_best].unique():

        temp = data_training[data_training[next_best] == each]

        class_count = temp[pred_attr].value_counts()

        # calculation of p_i and n_i, expected number of positives and negatives in each partition
        pi,ni,ct = 0.0, 0.0, class_count.index.values

        # for positive class
        if pos in ct:
            pi = float(class_count[1])

        # for negative class
        if neg in ct:
            ni = float(class_count[0])

        # T_i calculaton, which is of the total
        Ti = len(temp)


        # first coefficient of S
        first = (((positive * Ti / float(total)) - pi) ** 2) / float(positive * Ti / float(total))

        #second coefficient of S
        second = (((negative * Ti / float(total)) - ni) ** 2) / float(negative * Ti / float(total))


        S = S + first + second

    # calculating the chisquare criterion
    temp = chi2.cdf(x=S, df=len(data_training[next_best].unique()) - 1)
    p_value = 1 - temp

    # return True if the p-value is les than threshold, which means to expand
    if p_value < threshold_value:
        return False
    # threshold_value < p-value -> stop
    else:
        return True


# Function which will return the best attribute with least entropy - max gain
def next_best_choose(data_training, pred_attr, all_attr):
    data_training_length = len(data_training)

    # map for calculating the minimum entropy
    entropy_map = {}

    # Calculating entropy for each attribute
    a_tt = all_attr

    for single in a_tt:

        counts_list = []

        entropy_list = []

        for each in data_training[single].unique():

            temp = data_training[data_training[single] == each]

            total_count = len(temp)

            class_count = temp[pred_attr].value_counts()

            if len(class_count.index.values) != 2:
                entropy_list.append(0)

                counts_list.append(total_count)
                continue

            mul_val = (class_count[1] / float(total_count)) * np.log2(class_count[1] / float(total_count))

            mul_val = mul_val + (class_count[0] / float(total_count)) * np.log2(class_count[0] / float(total_count))

            entropy = -1 * mul_val

            counts_list.append(total_count)
            entropy_list.append(entropy)

        # initialize the final entropy
        final_entropy = 0

        # calculating the final entropy value by adding all the entropies for each attribute
        for i in range(len(entropy_list)):
            mul = counts_list[i] * entropy_list[i]

            final_entropy = final_entropy + mul

        # final entropy calculation
        final_entropy = final_entropy / data_training_length


        # appending all the entropies to a map
        entropy_map[single] = final_entropy




    # Calculating the best attribute which has minimum entropy
    best_attribute = min(entropy_map.items(), key=lambda x: x[1])

    return_val = best_attribute[0]

    return return_val



# Function which implements ID3
def id3_decision_tree(data_training, pred_attr, all_attr, threshold_value):
    global nodes_internal, node_leaf
    posi, nege = 1,0
    # if the sample only has all positives values, return TreeNode with data = 'True'
    if nege not in data_training[pred_attr].unique():
        node_leaf += 1
        return TreeNode()

    # if the sample only has all negative values, return TreeNode with data = 'False'
    if posi not in data_training[pred_attr].unique():
        node_leaf += 1
        return TreeNode(data='F', children=[])
    l_all_att,vall = len(all_attr),data_training[pred_attr].value_counts()
    # if there are no attributes left to split on, we build the node with positive or negative based on their count
    if l_all_att == 0:
        class_count = vall

        for i in [0,1]:
            if i == 1:
                pos = class_count[i]
            else:
                neg = class_count[i]

        # positive examples greater than negative
        if pos > neg:
            node_leaf += 1

            return TreeNode()

        # negative examples greater than positive
        else:
            node_leaf += 1

            return TreeNode(data='F', children=[])

    # chose the best attribute based for which gain is max
    next_best = next_best_choose(data_training, pred_attr, all_attr)

    # By chi-squared criterion we return True/ False depending on the pval
    stop_split = chisquare_split_stop(data_training, pred_attr, next_best, threshold_value)



    # if p-value calculated in chisquare_split is greater than the supposed limit - thres_limit
    # if it is True means p-value greater than threshold we will stop
    if stop_split == True:
        class_count = data_training[pred_attr].value_counts()

        pos = class_count[1]

        neg = class_count[0]

        # positive examples greater than negative
        if pos > neg:
            node_leaf += 1

            return TreeNode()
        # negative examples greater than negative
        else:
            node_leaf += 1

            return TreeNode(data='F', children=[])

    # setting the best attribute as the attribute to split on
    root = TreeNode(data=str(next_best + 1))

    nodes_internal += 1

    new_attr = list(all_attr)




    # removing the best attribute from the list of attributes to pass recursively

    new_attr.remove(next_best)
    # calling id3 recursively for each child of current node
    for each in range(5):
        new_example_data = data_training[data_training[next_best] == each + 1]

        #geeting the children
        temp = list(new_example_data.columns).index(next_best)
        new_example_data = new_example_data.drop(new_example_data.columns[temp], axis=1)

        # calling id3 for each child
        root.nodes[each] = id3_decision_tree(new_example_data, pred_attr, new_attr, threshold_value)

    return root

# This is to label encode the outputs to 0 and 1.
def label_encode(root, ex_point):
    true, false = 'T', 'F'

    if root.data == false:
        return 0

    if root.data == true:
        return 1

    return label_encode(root.nodes[ex_point[int(root.data) - 1] - 1], ex_point)


# loads Train and Test data
def load_data(ftrain, ftest):
    X_train, Y_train, Xtest = [], [], []
    with open(ftrain, 'rb') as f:
        reader = csv.reader(f)
        for row in reader:
            rw = map(int, row[0].split())
            X_train.append(rw)

    with open(ftest, 'rb') as f:
        reader = csv.reader(f)
        for row in reader:
            rw = map(int, row[0].split())
            Xtest.append(rw)

    Xtrain = pd.read_csv(ftrain, header=None, delim_whitespace=True)
    ftrain_label = ftrain.split('.')[0] + '_label.csv'
    with open(ftrain_label, 'rb') as f:
        reader = csv.reader(f)
        for row in reader:
            rw = int(row[0])
            Y_train.append(rw)
    Ytrain = pd.read_csv(ftrain_label, header=None, delim_whitespace=True)

    print('Data Loading: done')
    return Xtrain, Ytrain, Xtest


parser = argparse.ArgumentParser()
parser.add_argument('-p', required=True)
parser.add_argument('-f1', help='training file in csv format', required=True)
parser.add_argument('-f2', help='test file in csv format', required=True)
parser.add_argument('-o', help='output labels for the test dataset', required=True)
parser.add_argument('-t', help='output tree filename', required=True)

args = vars(parser.parse_args())

pval = args['p']
pval = float(pval)
Xtrain_name = args['f1']
Ytrain_name = args['f1'].split('.')[0] + '_labels.csv'  # labels filename will be the same as training file name but with _label at the end

Xtest_name = args['f2']
Ytest_predict_name = args['o']

tree_name = args['t']

Xtrain, Ytrain, Xtest = load_data(Xtrain_name, Xtest_name)

print("Training...")

data_training = pd.concat([Xtrain, Ytrain], axis=1, ignore_index=True)
pred_attr = len(data_training.columns) - 1

all_attr = list()

for each in data_training.columns:
    all_attr.append(each)

all_attr.remove(pred_attr)

start = timeit.default_timer()
s = id3_decision_tree(data_training, pred_attr, all_attr, pval)
stop = timeit.default_timer()

s.save_tree(tree_name)

print("Training took : ", stop - start)
print("Internal Nodes expanded : ", nodes_internal)
print("Leaf Nodes expanded : ", node_leaf)

print("Testing...")
Ypredict = []
for i in range(0, len(Xtest)):
    result = label_encode(s, Xtest[i])
    Ypredict.append([result])

with open(Ytest_predict_name, "wb") as f:
    writer = csv.writer(f)
    writer.writerows(Ypredict)

print("Output files generated")
