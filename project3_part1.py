#   Ryan Pauly
#   CS425 - Project 3
#
#   Project 3: K-Nearest-Neighbors
#
#
###################################################################

import pandas as pd
import numpy as np
from prettytable import PrettyTable
from collections import Counter


def find_distance(a, b):
    euclideanDistance = 0
    for i in range(len(a)):
        #print("\na[i] =", a[i])
        #print("\nb[i] =", b[i])
        euclideanDistance = euclideanDistance + ((a[i] - b[i])**2)
    return np.sqrt(euclideanDistance)


def find_mode(labels):
    return Counter(labels).most_common(1)[0][0]


def k_nearest_neighbor(data, test, k):

    #   For each variable we need to calculate the distance between itself and the data.
    neighbors = []              # A list of the distance and indices for each piece of data.
    gather_predictions = []     # A gathering of possible k predictions for a specific case
    predictions = []            # Final predictions which are of the most common or "mode" function on an element in
                                # gather predictions

    for index, observations in enumerate(data):
        #   print("\n observations = ", observations)
        #   print("\n obersations[0:9] =", observations[0:9])

        #   Here we will calculate the distance between the test and the data we wish to compare it with
        #   The test is actually the training data, which is what we're testing essentially.
        neighbors.clear()

        for i in range(len(test)):
            #print("\n test[i][0:9] = ", test[i][0:9])
            newDistance = find_distance(observations[1:9], test[i][1:9])
            # print("\n newDistance = ", newDistance)
            #   Append this newDistance result and its respective index to the neighbors list
            neighbors.append((newDistance, i))

        #   Next we need to sort from smallest to largest with respect to distance
        mySortedNeighbors = sorted(neighbors)
        #print("\nmySortedNeighbors = ", mySortedNeighbors)

        gather_predictions.clear()

        #   Gather the classifiers (predictions) from the first k entries of the sorted neighbors list
        for j in range(k):
            #   Gather the first k entry classifiers from the sorted neighbors list
            gather_predictions.append(test[mySortedNeighbors[j][1]][10])

        #print("\n gather_predictions = ", gather_predictions)

        myPrediction = find_mode(gather_predictions)
        #print("\nmyPrediction = ", myPrediction)

        predictions.append(myPrediction)

    TN = []
    TP = []
    FN = []
    FP = []
    #   2 == benign
    #   4 == malignant
    for i in range(len(predictions)):
        if predictions[i] == 2:
            if data[i][10] == 2:
                TN.append(1)  # True Negative
            else:
                FN.append(1)  # False Negative
        else:
            if data[i][10] == 2:
                FP.append(1)  # False Positive
            else:
                TP.append(1)  # True Positive

    num_TN = len(TN)
    num_TP = len(TP)
    num_FN = len(FN)
    num_FP = len(FP)

    accuracy = ((num_TN + num_TP) / (num_TN + num_TP + num_FN + num_FP)) * 100
    TPR = num_TP / ( num_TP + num_FN)
    PPV = num_TP / (num_TP + num_FP)
    TNR = num_TN / (num_TN + num_FP)
    F_1_Score = 2 * PPV * TPR / (PPV + TPR)

    print(" Accuracy = ", accuracy)
    print(" TPR = ", TPR)
    print(" PPV = ", PPV)
    print(" TNR = ", TNR)
    print(" F_1_Score = ", F_1_Score)

    #   Create Confusion Matrix

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

    #print(myData.dtypes)
    #print(myData.shape)

    #   Shuffle the rows of the DataFrame->numpyArray

    #print("myData = \n", myData)
    myData = myData.to_numpy()

    np.random.shuffle(myData)
    #   np.savetxt("shuffledData.csv", myData, delimiter=",")

    #   A good split I found was to separate training to 70%, validation to 10%, and testing for 20%
    #   Total rows of data after cleaning = 683.

    training = myData[:476, :]
    validation = myData[476:544, :]
    testing = myData[544:, :]

    #   print("\n length of training = ", len(training))
    #   print("\n length of validation = ", len(validation))
    #   print("\n length of testing = ", len(testing))

    k_values = [2, 3, 4, 5, 6, 7, 8, 17, 33]  # We will use the validation data to determine the best k value

    #   k_nearest_neighbor(validation, training, 3)
    accuracyCheck = 0
    k_bestAccuracy_index = 0

    print("\n\nCROSS-VALIDATION:")
    for i in range(len(k_values)):
        print("\nFor k = ", k_values[i])
        check = k_nearest_neighbor(validation, training, k_values[i])
        if check > accuracyCheck:
            accuracyCheck = check   # update the max accuracy
            k_bestAccuracy_index = i             # save the index which had the best k value

    print("\n\nThe best k value is ", k_values[k_bestAccuracy_index], "\n")

    print("Now with the testing and training datasets and the best found k value: k = ", k_values[k_bestAccuracy_index])
    k_nearest_neighbor(testing, training, k_values[k_bestAccuracy_index])
