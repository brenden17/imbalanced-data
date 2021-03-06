""" Under sampling and over sampling on imbalanced data
Dataset : http://archive.ics.uci.edu/ml/datasets/Balance+Scale
Ref : http://bi.snu.ac.kr/Publications/Conferences/Domestic/KIISE2013f_KMKim.pdf
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import SVC
from sklearn.cross_validation import cross_val_score, train_test_split
from sklearn.metrics import classification_report

def get_fullpath(filename):
    return os.path.abspath(os.path.join(os.path.dirname(__file__), filename))

def read_file():
    return pd.read_csv(get_fullpath('balance-scale.data'), delimiter=',',
                       names=['class', 'lweight', 'ldist', 'rweight', 'rdist'])

def analysis_data():
    df = read_file()
    X_outlier = df[df['class'] == 'B'] # .ix[idx]
    X_train = df[df['class'] == 'L']

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter(X_outlier['lweight'], X_outlier['ldist'],
               X_outlier['rweight'], c='r')
    ax.scatter(X_train['lweight'], X_train['ldist'],
               X_train['rweight'], c='b')
    ax.set_xlabel('Mileage')
    ax.set_ylabel('Transmission')
    ax.set_zlabel('Price')
    plt.show()

def load_original_data():
    rawdata = read_file()
    le = LabelEncoder()
    X = rawdata.icol(range(1, 5)).values
    y = le.fit_transform(rawdata['B'].values)
    return X, y

def load_undersampling():
    rawdata = read_file()
    n_sample = rawdata[rawdata['class'] == 'B'].shape[0]
    B = rawdata[rawdata['class'] == 'B']
    L = rawdata[rawdata['class'] == 'L'][:n_sample]
    R = rawdata[rawdata['class'] == 'R'][:n_sample]
    d = pd. concat([B, L, R])
    le = LabelEncoder()
    X = d.icol(range(1, 5)).values
    y = le.fit_transform(d['class'].values)
    return X, y

def load_sampling():
    size = 200
    rawdata = read_file()
    n_sample = rawdata[rawdata['class'] == 'B'].shape[0]
    idx = np.random.randint(0, n_sample, size)
    B = rawdata[rawdata['class'] == 'B'].iloc[idx]

    n_sample = rawdata[rawdata['class'] == 'L'].shape[0]
    idx = np.random.randint(0, n_sample, size)
    L = rawdata[rawdata['class'] == 'L'].iloc[idx]

    n_sample = rawdata[rawdata['class'] == 'R'].shape[0]
    idx = np.random.randint(0, n_sample, size)
    R = rawdata[rawdata['class'] == 'R'].iloc[idx]

    df = pd.concat([B, L, R])

    le = LabelEncoder()
    X = df.icol(range(1, 5)).values
    y = le.fit_transform(df['class'].values)
    return X, y

def load_data_with_SMOTE():
    rawdata = read_file()
    size = 150
    small = rawdata[rawdata['class'] == 'B']
    n_sample = small.shape[0]
    idx = np.random.randint(0, n_sample, size)
    X = small.iloc[idx, range(1, 5)].values
    y = small.iloc[idx, 0].values
    knn = NearestNeighbors(n_neighbors=2)
    knn.fit(X)
    _d, i = knn.kneighbors(X)
    idx2 = i[:, 1]
    diff = X - X[idx2]
    X = X + np.random.random(4) * diff
    B = np.concatenate([np.transpose(y[np.newaxis]), X], axis=1)
    B = pd.DataFrame(B)

    n_sample = rawdata[rawdata['class'] == 'L'].shape[0]
    idx = np.random.randint(0, n_sample, size)
    L = rawdata[rawdata['class'] == 'L'].iloc[idx]

    n_sample = rawdata[rawdata['class'] == 'R'].shape[0]
    idx = np.random.randint(0, n_sample, size)
    R = rawdata[rawdata['class'] == 'R'].iloc[idx]

    d = np.concatenate([B.values, L.values, R.values])

    le = LabelEncoder()
    X = d[:, 1:5]
    y = le.fit_transform(d[:, 0])
    return X, y

def create_learners():
    return [ExtraTreesClassifier(), SVC(), LogisticRegression()]

def analysis(load_function):
    X, y = load_function()
    X_train, X_test, Y_train, Y_test, = \
                        train_test_split(X, y,
                                         test_size=0.3, random_state=42)
    size = 3
    learners = create_learners()
    for learner in learners:
        print '%s -\t\t %f' % (learner.__class__.__name__,
                           sum(cross_val_score(learner, X, y, cv=size)) / size)
        learner.fit(X_train, Y_train)
        Y_predict = learner.predict(X_test)
        print learner.__class__.__name__
        print classification_report(Y_test, Y_predict)

if __name__ == '__main__':
    print "=================== Original Data ==================="
    analysis(load_original_data)
    print "=================== Under Sampling ==================="
    analysis(load_undersampling)
    print "=================== Sampling ==================="
    analysis(load_sampling)
    print "=================== SMOTE ==================="
    analysis(load_data_with_SMOTE)