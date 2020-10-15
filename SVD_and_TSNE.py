import os
import numpy as np
import pandas as pd
import random as rn

# Visualization libraries
import pydotplus
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style({"axes.facecolor": ".95"})

# Modeling and Machine Learning
from IPython.display import Image
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
#from sklearn.externals.six import StringIO
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz

seed = 1234
np.random.seed(seed)
rn.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)


"""
The train and test MNIST datasets are combined and the labels removed
"""
train_df = pd.read_csv('data/mnist_train.csv')
val_df = pd.read_csv('data/mnist_test.csv')

MNIST_df = train_df.append(val_df)
d = MNIST_df.drop("label", axis=1)

tsvd_n_components = 784

"""
TruncatedSVD from sklearn is used to reduce the dimension of each sample from 784 to 50
"""
tsvd = TruncatedSVD(n_components=50).fit_transform(d)

"""
TSNE from sklearn is usec to reduce dimensionality from 50 to 3
"""
tsne = TSNE(n_components=3)
transformed = tsne.fit_transform(tsvd)


"""
The data is stored in a dataframe and labels are re-added
"""

tsne_data = pd.DataFrame(transformed, columns=['component1', 'component2', 'component3'])
#tsne_test = pd.DataFrame(transformed[len(train_df):], columns=['component1', 'component2', 'component3'])
tsne_data['label'] = MNIST_df['label']
# plt.figure(figsize=(14, 14))
# plt.title(f"Visualization of t-SNE results on the MNIST Dataset\n\
# Amount of datapoints: {len(tsne_train)}", fontsize=24, weight='bold')
# sns.scatterplot("component1", "component2",
#                 data=tsne_train, hue=train_df['label'],
#                 palette="Set1", legend="full")
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)
# plt.xlabel("Component 1", fontsize=16)
# plt.ylabel("Component 2", fontsize=16)
# plt.legend(fontsize=16)
# plt.show()
# print('train', tsne_train)
# print('test', tsne_test)

tsne_data.to_csv('TSNE_all_untouched')

