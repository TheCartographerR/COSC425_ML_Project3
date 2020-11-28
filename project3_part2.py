#   Ryan Pauly
#   CS425 - Project 3
#
#   Project 3: Decision Trees
#
# use a classifier decision tree to predict benign/malignant using cross-validation
#
# The user should be able to select the impurity measure they wish to use:
#     these impurity measure methods include: gini index, misclassification error, and entropy
#
# The user is also able to enter a maximum depth for the decision tree, as well as an impurity threshold
#     the map depth is within the range 2 to 10
#     and the impurity threshold is really anywhere between 0 and 1 (where 0 == perfect purity)
#
###################################################################
import pandas as pd
import numpy as np
import math
from prettytable import PrettyTable

depthTracker = -1


#   Impurity Measures:

#   Entropy
def find_entropy(rows):
    totalRowsAtSplit = 0.0
    for row in rows:
        totalRowsAtSplit += len(row)

    entropy = 0.0

    for split in rows:
        #   A split (left or right) may be empty, and if it is, we just need to skip it
        if len(split) == 0:
            continue

        p = [row[9] for row in split].count(2.0) / len(split)
        p_not = [row[9] for row in split].count(4.0) / len(split)
        # print("\n\np = ", p)
        # print("p_not = ", p_not)

        #   if either are a 1 numpy.log2 gets 0 but the value is actually huge like 0.0000000000? And so there is
        #   some kind of bug? At least from my research about it, but I could be wrong, either way, I chose to
        #   just skip this iteration if this was encountered.

        temp = 0
        if p == 0:
            if p_not == 0:
                temp = 0
            else:
                temp = (-1 * p_not * math.log2(p_not))
        else:
            if p_not == 0:
                temp = (-1 * p * math.log2(p))
            else:
                temp = (-1 * p * math.log2(p)) + (-1 * p_not * math.log2(p_not))

        # print("temp = ", temp)
        entropy += temp * (len(split) / totalRowsAtSplit)

        # print("entropy = ", entropy)
    # print("entropy = ", entropy)
    return entropy


#   Gini
def find_gini(rows):
    totalRowsAtSplit = 0.0
    for row in rows:
        # print("row =", row)
        totalRowsAtSplit += len(row)

    gini_impurity = 0.0

    for split in rows:
        #   A split (left or right) may be empty, and if it is, we just need to skip it
        if len(split) == 0:
            continue

        p = [row[9] for row in split].count(2.0) / len(split)
        p_not = [row[9] for row in split].count(4.0) / len(split)

        temp = (p ** 2) + (p_not ** 2)
        #   print("temp = ", temp)

        gini_impurity += (1 - temp) * (len(split) / totalRowsAtSplit)

    #   print("gini_impurity = ", gini_impurity)
    return gini_impurity


#   Misclassification Error
def find_misclassificationError(rows):
    totalRowsAtSplit = 0.0
    for row in rows:
        totalRowsAtSplit += len(row)

    misclassification_error = 0.0

    for split in rows:
        #   A split (left or right) may be empty, and if it is, we just need to skip it
        if len(split) == 0:
            continue

        p = [row[9] for row in split].count(2.0) / len(split)
        p_not = [row[9] for row in split].count(4.0) / len(split)

        temp = 1 - max(p, p_not)
        # print("\n\ntemp = ", temp)

        misclassification_error += (temp) * (len(split) / totalRowsAtSplit)

    #   print("misclassification_error = ", misclassification_error)
    return misclassification_error


#   Splits the data passed into it into a left and right based off the passed in conditional
def split_partition(data, index_of_split, conditional_value):
    left_split = []  # Less than the conditional
    right_split = []  # Greater than or equal to the conditional
    #   Left node takes values that are less than the determined conditional value of the split.
    for row in data:
        if row[index_of_split] < conditional_value:
            left_split.append(row)
        else:
            #   Greater or equal to the conditional value
            right_split.append(row)

    # print("\n\nleft_split = ", left_split)
    # print("right_split = ", right_split)

    # return a list of lists for the left and right split rows of data

    return left_split, right_split


def find_best_split(data, user_impurity_measure):
    #   This function is for finding the best possible split of the passed in data based off the exhaustive impurity
    #   measure (entropy, gini, and misclassification error) provided by the user, the best impurity being a 0 (pure)
    #   Where we want the purest possible split

    split_index = 0
    conditional_value_for_split = 0
    rows = 0
    find_lowest_impurity = 100000000000

    #   First traverse through each feature excluding the classifier
    for cell_feature in range(len(data[0]) - 1):
        #   I have to find what the best feature to use to determine the conditional for a split
        #   To do this, look at a specific "cell" value and try it as the conditional in the split_partition function
        for row in data:  # going down each row of this cell_feature and checking its item value
            conditional_to_test = row[cell_feature]
            temp_rows = split_partition(data, cell_feature, conditional_to_test)

            impurity = 0
            #   Next use the user's chosen user impurity method
            # Entropy
            if user_impurity_measure == 0:
                impurity = find_entropy(temp_rows)
            # Gini Index
            if user_impurity_measure == 1:
                impurity = find_gini(temp_rows)
            # Misclassification Error
            if user_impurity_measure == 2:
                impurity = find_misclassificationError(temp_rows)

            #   print("impurity = ", impurity)
            if impurity <= find_lowest_impurity:
                rows = temp_rows
                split_index = cell_feature  # the feature we'll use to determine the split
                conditional_value_for_split = conditional_to_test
                find_lowest_impurity = impurity  # It will be forced to go smaller or stay the same

    #   Return a dictionary object with information for a node of the tree and its respective split
    myNode = {
        "impurity": find_lowest_impurity,
        "feature": split_index,
        "conditional": conditional_value_for_split,
        "rows": rows
    }

    return myNode


#   DECISION TREE LEAF NODE
def make_leaf(rows):
    find_most_common_classifier = []
    for row in rows:
        if len(row) == 0:
            continue
        find_most_common_classifier.append(row[9])

    if len(find_most_common_classifier) == 0:
        return
    else:
        return max(set(find_most_common_classifier), key=find_most_common_classifier.count)


#   DECISION TREE RECURSION (BUILDING CHILD NODES)
def make_children(node, depth, user_max_depth, user_impurity_choice, user_impurity_threshold):
    #   The node is a dictionary object containing the left and right split data as well as the feature that was used
    #   along with the split conditional value used to split the data into a left and right partition
    #   Depth will keep track of the depth for recursive purposes

    global depthTracker

    child_node_data_limit = 40  # We want to give a limit of how many child nodes it can have before a leaf node is made
    # Essentially, if we have 40 or fewer rows of data then we want to just make a leaf
    # and estimate the classifier with what we have rather than making more child nodes
    # with that small amount of data

    if node["rows"] == 0:
        return

    left_of_split, right_of_split = node["rows"]  # store the left and right split lists of rows respectively

    node["left_branch"] = 0
    node["right_branch"] = 0

    #   Check user impurity threshold and if it is less than or equal to the threshold, make it a leaf.
    if node["impurity"] <= user_impurity_threshold:
        node["left_branch"] = make_leaf(left_of_split)
        node["right_branch"] = make_leaf(right_of_split)

    #   First check if the max_depth set by the user has been reached
    if depthTracker == user_max_depth:
        #   Make the left and right branches leaf nodes:
        node["left_branch"] = make_leaf(left_of_split)  # Take the left of split data and use it to make a leaf
        node["right_branch"] = make_leaf(right_of_split)  # Take the right of split data and use it to make a leaf
        return

    #   Check to see if either the left or right split of the rows is empty
    if not left_of_split or not right_of_split:
        #   One of the
        #   Both the left and right branch of the node are thus set equal to a leaf with both of their content
        node["left_branch"] = make_leaf(left_of_split + right_of_split)
        node["right_branch"] = make_leaf(left_of_split + right_of_split)
        return

    # Increment the depthTracker, since the tree is unbalanced we just increment whenever one or the other makes
    # a new child node (there are more rows than the cutoff of child_node_data_limit (greater than 20 rows)
    if (len(left_of_split) > child_node_data_limit) or (len(right_of_split) > child_node_data_limit):
        depthTracker += 1

    # Left Child side first:
    if not len(left_of_split) <= child_node_data_limit:
        #   First find the best split of this left split's rows of data
        node["left_branch"] = find_best_split(left_of_split, user_impurity_choice)
        #   Then, recursively call make_children on the returned value of node["left_branch"]
        depth += 1
        make_children(node["left_branch"], depth, user_max_depth, user_impurity_choice, user_impurity_threshold)

    else:
        node["left_branch"] = make_leaf(left_of_split)

    if not len(right_of_split) <= child_node_data_limit:
        #   Now the same for the right branch
        node["right_branch"] = find_best_split(right_of_split, user_impurity_choice)
        #   Then, recursively call make_children on the returned value of node["left_branch"]
        depth += 1
        make_children(node["right_branch"], depth, user_max_depth, user_impurity_choice, user_impurity_threshold)
    else:
        node["right_branch"] = make_leaf(right_of_split)


def plant_the_acorn(data, user_max_depth, user_impurity_choice, user_impurity_threshold):
    global depthTracker
    depthTracker = 0

    root_node = find_best_split(data, user_impurity_choice)

    make_children(root_node, depthTracker, user_max_depth, user_impurity_choice, user_impurity_threshold)

    print("\ndepthTracker = ", depthTracker)
    return root_node


def find_tree_predictions(node, testing_set_rows):
    #   First check the testing_set_row's feature and if its less than the conditional
    #   that was used to split this node
    if not isinstance(node, dict):
        return

    if testing_set_rows[node["feature"]] < node["conditional"]:
        #   Check if it is a child node or a leaf
        if not isinstance(node["left_branch"], dict):
            #   Found a leaf node because it is not a dictionary obj
            #   Thus, return the leaf node
            return node["left_branch"]

        else:
            #   Found a child node (it has branches)
            #   Recursively call find_tree_predictions on this child node
            return find_tree_predictions(node["left_branch"], testing_set_rows)

    else:
        #  Now the right branch
        #   First check if it is a child node or a leaf node
        if not isinstance(node["right_branch"], dict):
            #   LEAF NODE
            return node["right_branch"]

        else:
            #   A CHILD NODE
            return find_tree_predictions(node["right_branch"], testing_set_rows)


def tree_statistics(training, data_to_test, max_depth, user_impurity_measure, threshold, print_option):
    myDecisionTree = plant_the_acorn(training, max_depth, user_impurity_measure, threshold)
    myPredictions = []
    actual = []

    # print("myDecisionTree Root = ", myDecisionTree)
    # print("depthTracker = ", depthTracker)
    for row in data_to_test:
        temp = find_tree_predictions(myDecisionTree, row)
        myPredictions.append(temp)
        actual.append(row[9])

    if print_option == 1:
        print("actual = ", actual)
        print("myPredictions = ", myPredictions)

    TN = []
    TP = []
    FN = []
    FP = []
    #   2 == benign
    #   4 == malignant
    for i in range(len(myPredictions)):
        if myPredictions[i] == 2:
            if actual[i] == 2:
                TN.append(1)  # True Negative
            else:
                FN.append(1)  # False Negative
        else:
            if actual[i] == 2:
                FP.append(1)  # False Positive
            else:
                TP.append(1)  # True Positive

    num_TN = len(TN)
    num_TP = len(TP)
    num_FN = len(FN)
    num_FP = len(FP)

    accuracy = ((num_TN + num_TP) / (num_TN + num_TP + num_FN + num_FP)) * 100
    TPR = num_TP / (num_TP + num_FN)
    PPV = num_TP / (num_TP + num_FP)
    TNR = num_TN / (num_TN + num_FP)
    F_1_Score = 2 * PPV * TPR / (PPV + TPR)

    if print_option == 1:
        print(" Accuracy = ", accuracy)
        print(" TPR = ", TPR)
        print(" PPV = ", PPV)
        print(" TNR = ", TNR)
        print(" F_1_Score = ", F_1_Score)

        # Create Confusion Matrix

        table = PrettyTable()
        table.field_names = ["...", "Predicted Class", 'Benign', 'Malignant']
        table.add_row((["True", "Benign", num_TN, num_FP]))
        table.add_row([" Class", "Malignant", num_FN, num_TP])
        print(table)

    return accuracy


if __name__ == "__main__":
    fileName = "breast-cancer-wisconsin.data"

    #   Read in the CSV file
    myData = pd.read_csv(fileName, header=None)

    #   There are '?' symbols in some cells of the .data file (CSV).
    #   We must handle this somehow.

    #   Replace '?' with np.nan
    myData = myData.replace(to_replace="\?", value="", regex=True)
    myData = myData.replace("", np.nan).astype(float)

    #   Next drop the rows which contain a np.nan value, essentially deleting rows with incomplete data.
    myData = pd.DataFrame.dropna(myData, axis=0, how='any', thresh=None, subset=None, inplace=False)

    #   Shuffle the rows of the DataFrame->numpyArray
    myData = myData.to_numpy()
    np.random.shuffle(myData)

    #   We don't need the ID value in the first column, so we'll just go ahead and chop it out.
    #   We didn't need it in part 1, but just adjusted for it later on, to avoid issues in part 2 I'll take care
    #   of this now.
    myData = myData[:, 1:]

    #   np.savetxt("shuffledData.csv", myData, delimiter=",")

    #   A good split I found was to separate training to 70%, validation to 10%, and testing for 20%
    #   Total rows of data after cleaning = 683.

    training = myData[:476, :]
    validation = myData[476:544, :]
    testing = myData[544:, :]

    impurity_methods = ["entropy", "gini", "misclassification error"]
    k_depths = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    thresh_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    max_depth = 10
    user_impurity_measure = 1
    threshold = 0.8
    #print("User Input Method: \n")

    #tree_statistics(training, testing, max_depth, user_impurity_measure, threshold, 1)

    #   CROSS VALIDATION WITH EACH IMPURITY METHOD
    best_accuracy = 0
    best_impurity = 0
    best_depth = 0
    best_thresh = 0

    print("\n*******************************************************************************************************")
    print("\nCROSS VALIDATION: \n")

    # for i in range(len(impurity_methods)):
    #     #   Find which impurity measure has the best accuracy
    #     temp = tree_statistics(training, validation, max_depth, i, threshold, 0)
    #     print("\n" + impurity_methods[i] + " accuracy = " + str(temp))
    #     if temp > best_accuracy:
    #         best_accuracy = temp
    #         best_impurity = i

    for thresh in range(len(thresh_list)):
        for depth in range(len(k_depths)):
            for i in range(len(impurity_methods)):
                #   Find the best accuracy
                temp = tree_statistics(training, validation, k_depths[depth], i, thresh_list[thresh], 0)
                #print("\n" + impurity_methods[i] + " accuracy = " + str(temp))
                if temp > best_accuracy:
                    best_accuracy = temp
                    best_impurity = i
                    best_depth = depth
                    best_thresh = thresh

    print("\nThe best impurity measure is: ", impurity_methods[best_impurity])
    print("The best k_depth is: ", k_depths[best_depth])
    print("The best threshold is: ", thresh_list[best_thresh])
    print("Which found an accuracy of = ", best_accuracy)

    print("\n\nNow with testing data and the best impurity measure: ")
    tree_statistics(training, testing, k_depths[best_depth], best_impurity, thresh_list[best_thresh], 1)

    # entropy_accuracy = tree_statistics(training, testing, max_depth, 0, threshold, 1)
    # gini_accuracy = tree_statistics(training, testing, max_depth, 1, threshold, 1)
    # misclass_accuracy = tree_statistics(training, testing, max_depth, 2, threshold, 1)
    #
    # print("entropy_accuracy = ", entropy_accuracy)
    # print("gini_accuracy = ", gini_accuracy)
    # print("Misclass_accuracy = ", misclass_accuracy)

    # myDict = find_best_split(validation, user_impurity_measure, threshold)

    # print("myDict = ", myDict)

    # plant_the_acorn(training, max_depth, user_impurity_measure, threshold)
