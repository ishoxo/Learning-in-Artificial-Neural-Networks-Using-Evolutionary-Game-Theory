import os
import numpy as np
import pandas as pd
import random as rn
import matplotlib
import matplotlib.pyplot as plt
train_df = pd.read_csv('data/mnist_train.csv')
val_df = pd.read_csv('data/mnist_test.csv')

MNIST_df = train_df.append(val_df)
#MNIST_df = MNIST_df.head(1000)


d = train_df.drop("label", axis=1)
from sklearn.decomposition import TruncatedSVD
tsvd_n_components = 783


svd = TruncatedSVD(n_components=tsvd_n_components)
svd.fit(d)

explained_variance = list(svd.explained_variance_)
percentage_explained_var = [x / sum(explained_variance) for x in explained_variance]
cum_var_explained = np.cumsum(percentage_explained_var)
print(explained_variance)
plt.figure(1, figsize=(6, 4))

plt.clf()
plt.plot(cum_var_explained, linewidth=2)
plt.axis('tight')
plt.grid()
plt.xlabel('n_components')
plt.ylabel('Cumulative_explained_variance')
plt.show()