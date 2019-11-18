#!/usr/bin/env python3
import numpy as np
from sklearn.model_selection import  cross_val_score
from sklearn.cluster import KMeans
from sklearn.linear_model import  LogisticRegression
from sklearn.metrics import silhouette_score, homogeneity_completeness_v_measure
from sklearn.preprocessing import OneHotEncoder


""" Assignment 4: clustering for fun and profit
    See <https://snlp2019.github.io/a4/> for instructions.
    Author: Chen, Pin-Zhen 
    
"""

def cluster(unlab, labeled, labels, k=7):
    """ Cluster given unlabeled data using k-means,
        and evaluate on the labeled date set.

    Arguments:
        unlab    An Nx2 array, columns are first (f1) and second (f1) formants.
        labeleed An Mx2 array where columns are f1 and f2
        labels   An sequence with M labels (vowels)
    """
    model = KMeans(n_clusters=k , random_state=32)
    cluster_labels = model.fit_predict(unlab)

    # calculates the silhouette score, and prints it
    s = silhouette_score(unlab, labels=cluster_labels)
    print("Silhouette_score: ",s)

    # calculates and prints the homogeneity, completeness and v-measure on the labeled data set
    labs_labeled = model.fit_predict(labeled)
    h_c_v = homogeneity_completeness_v_measure(labels, labs_labeled)
    print("Homogeneity_completeness_v_measure: ", h_c_v)

    return model
def classify(labeled, labels, kmeans=None):
    """ classify without/with clustering

    Atguments:
        labeled   the f1 and f2 values for the labeled data set
        labels    the corresponding labels 
        kmeans    a trained k-means model (from exercise 1)
    """
    ### 4.2 Classification
    f = open('f1score.txt', 'a+')
    log = LogisticRegression(random_state=17, C=1, class_weight='balanced', penalty='l1')
    clf = log

    f1 = cross_val_score(clf, labeled, labels, cv=5, scoring='f1_macro')
    print('F1 4.2: '+ str(np.mean(f1))+ "  ")

    ### 4.3 Semi-supervised classification with cluster labels

    # cluster one-hot encode
    onehotencoder = OneHotEncoder()
    cluster = onehotencoder.fit_transform(kmeans.labels_.reshape(-1,1)).toarray()

    # append cluster one-hot
    num_append =cluster.shape[1]
    features = np.zeros(( labeled.shape[0], labeled.shape[1]+num_append))
    features[:,0:2] = labeled
    features[:,2:] = cluster
    f1 = cross_val_score(clf, features, labels, cv=5, scoring='f1_macro')
    print('F1 4.3: '+ str( np.mean(f1))+ "  ")
    clf.fit(labeled, labels)


    ### 4.4 Semi-supervised classification using distances from cluster centers
    num_append = len(kmeans.cluster_centers_)
    feat_dist = np.zeros((features.shape[0], features.shape[1] + num_append))
    feat_dist[:, 0:features.shape[1]] = features

    for i, (cx, cy) in enumerate(kmeans.cluster_centers_):
        mean_distance = k_mean_distance(labeled, cx, cy)
        feat_dist[:, features.shape[1]+i] = mean_distance
    f1 = cross_val_score(clf, feat_dist, labels, cv=5, scoring='f1_macro')
    print('F1 4.4: '+ str(np.mean(f1)))

def k_mean_distance(data, cx, cy):
    # Calculate Euclidean distance for each data point assigned to centroid
    distance = [np.sqrt((x-cx)**2+(y-cy)**2) for (x, y) in data]
    return distance

if __name__ == '__main__':
    # handle file reading and call the functions above.
    vowel_unlab= np.loadtxt("vowels-unlabeled.txt", dtype=float , skiprows=1)
    vowel_lab = np.loadtxt("vowels-labeled.txt", dtype=float, usecols=(0,1),skiprows=1)
    labels = np.loadtxt("vowels-labeled.txt", dtype=str, usecols=2, skiprows=1)
    kmeans = cluster(vowel_unlab,vowel_lab,labels)
    classify(vowel_lab, labels, kmeans)