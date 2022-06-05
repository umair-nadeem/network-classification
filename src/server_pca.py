import os
import pandas as pd
import numpy as np
from sklearn.decomposition import *
from sklearn.preprocessing import *

os.chdir('/home/umair/PycharmProjects/thesis/files/classification_csv')

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

"""

lbd = LabelBinarizer()
train_dest = lbd.fit_transform(train['destination'])
test_dest = lbd.fit_transform(test['destination'])

svdmodel_destination = TruncatedSVD(n_components=500)

train_dest_red = svdmodel_destination.fit_transform(train_dest)
train_destination = pd.DataFrame(train_dest_red)
train_destination.to_csv('destination_train.csv', index=False)

test_dest_red = svdmodel_destination.fit_transform(test_dest)
test_destination = pd.DataFrame(test_dest_red)
test_destination.to_csv('destination_test.csv', index=False)
"""

lbs = LabelBinarizer()
train_src = lbs.fit_transform(train['source'])
test_src = lbs.fit_transform(test['source'])

svdmodel_source = TruncatedSVD(n_components=140)

train_src_red = svdmodel_source.fit_transform(train_src)
train_source = pd.DataFrame(train_src_red)
train_source.to_csv('source_train.csv', index=False)

test_src_red = svdmodel_source.fit_transform(test_src)
test_source = pd.DataFrame(test_src_red)
test_source.to_csv('source_test.csv', index=False)


