# Semi-supervised-learning-for-clustering-vowels
Semi-supervised learning for clustering vowels

Package: sklearn, numpy

This exercise is about clustering, evaluation of clustering results,
and using clustering to improve classification.
We will cluster vowels based on their first and second
[formants](https://en.wikipedia.org/wiki/Formant),
and evaluate the results using two different methods,
one intrinsic measure,
[silhouette score](https://en.wikipedia.org/wiki/Silhouette_(clustering)),
which evaluates the results only based on clusering configuration,
and another set of measures
[homogeniety, completeness and V-measure](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.homogeneity_completeness_v_measure.html),
which require gold-standard labels.
We will also use the clustering
(learned on a larger unlabeled data set)
for improving classification results (on a smaller labeled data set).
