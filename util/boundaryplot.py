__author__ = 'chrisbillovits'

''' A module to print the decision boundary of a classifier. '''

# For display using port forwarding

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('GTKAgg')

from sklearn.linear_model import LogisticRegression
from sklearn import datasets
import pylab
import numpy as np
from sklearn.decomposition import TruncatedSVD

n_classes = 3
plot_colors = 'bry'
plot_step = 0.02
lsa = TruncatedSVD(n_components = 2, algorithm = 'arpack', random_state=42)


def plot_boundary(clf, X, y):
    ''' Plots the three-class boundary learned by a classifier.'''
    decision = {'ENTAILMENT': 0, 'CONTRADICTION' : 1, 'NEUTRAL' : 2}
    decision_list = ['ENTAILMENT', 'CONTRADICTION', 'NEUTRAL']
    Z = clf.fit(X, y)
        
    
    X, _ = clf._pre_transform(X, y)

    clf = clf.steps[-1][1]
     
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step / 2))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    Z = [[decision[entry] for entry in row] for row in Z]

    cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
    
    plt.xlabel("Compressed Feature 1")
    plt.ylabel("Compressed Feature 2")
      
    y = [decision[entry] for entry in y]

    for i, color in zip(range(n_classes), plot_colors):
        idx = [j for j in range(len(y)) if y[j] == i]
        
        plt.scatter(X[idx, 0], X[idx, 1], c = color, label = decision_list[i],
                    cmap = plt.cm.Paired)
    
    plt.axis("tight")
    plt.legend()
    
    plt.savefig('output/plot.png')
