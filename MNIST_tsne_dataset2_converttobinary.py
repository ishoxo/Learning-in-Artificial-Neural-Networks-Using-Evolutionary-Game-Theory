import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import time
from torch.utils.data import Dataset as Dataset

train_size = 50000
val_size = 10000
test_size = 10000
MNIST_data = pd.read_csv('TSNE_all_untouched')
MNIST_train = MNIST_data[:train_size]
MNIST_val = MNIST_data[train_size: train_size + val_size]
MNIST_test = MNIST_data[train_size + val_size:]

def transform_data(MNIST_dataframe):
    comp1 = MNIST_dataframe['component1'].to_numpy()
    comp2 = MNIST_dataframe['component2'].to_numpy()
    comp3 = MNIST_dataframe['component3'].to_numpy()
    #label = MNIST_dataframe['label'].to_numpy()

    ma1 = max(comp1)
    mi1 = min(comp1)
    ma2 = max(comp2)
    mi2 = min(comp2)
    ma3 = max(comp3)
    mi3 = min(comp3)


    range1 = ma1 - mi1
    range2 = ma2 - mi2
    range3 = ma3 - mi3
    ranges = [range1, range2, range3]
    mins = [mi1, mi2, mi3]



    Row_list = []
    # Iterate over each row
    for index, rows in MNIST_dataframe.iterrows():
        # Create list for the current row
        my_list = [rows.component1, rows.component2, rows.component3, rows.label]
        # append the list to the final list
        Row_list.append(my_list)

    binary_row_list = []
    row_labels = []
    for i in range(len(MNIST_dataframe)):
        bin_row = []
        data_point = Row_list[i]
        for j in range(len(data_point) - 1):
            data_range = ranges[j]
            comp = data_point[j]
            comp = comp - mins[j]
            if comp > (data_range/2):
                bin_row.append(1)
            else:
                bin_row.append(0)
            if (comp % (data_range/2)) > (data_range/4):
                bin_row.append(1)
            else:
                bin_row.append(0)
            if (comp % (data_range/2)) % (data_range/4) > (data_range/8):
                bin_row.append(1)
            else:
                bin_row.append(0)
        binary_row_list.append(np.asarray(bin_row))
        row_labels.append(data_point[-1])

    return binary_row_list, row_labels



X_train, y_train = transform_data(MNIST_train)
X_val, y_val = transform_data((MNIST_val))
X_test, y_test = transform_data((MNIST_test))



MNIST_train_binary_dic = {'X': X_train, 'y': y_train}

MNIST_val_binary_dic = {'X': X_val, 'y': y_val}

MNIST_test_binary_dic = {'X': X_test, 'y': y_test}

MNIST_binary_train = pd.DataFrame(MNIST_train_binary_dic)
MNIST_binary_val = pd.DataFrame(MNIST_val_binary_dic)
MNIST_binary_test = pd.DataFrame(MNIST_test_binary_dic)


# MNIST_binary_train.to_csv('MNIST_binary_train')
