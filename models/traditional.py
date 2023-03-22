# Acknowledgement:
# Some of the codes are adapted from Loglizer project(https://github.com/logpai/loglizer). 

import numpy as np
from sklearn.metrics import precision_recall_fscore_support


def metrics(y_pred, y_true):
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    return precision, recall, f1



seq_level_data = np.load('./data/BGL/agg/BGL_fasttext_template_tfidf_50d_agg.npz', allow_pickle=True)

x_train = seq_level_data["x_train"]
y_train = seq_level_data["y_train"]
x_test = seq_level_data["x_test"]
y_test = seq_level_data["y_test"]

##############################
#  Logistic Regression
##############################

from sklearn.linear_model import LogisticRegression

class LR(object):

    def __init__(self, penalty='l2', C=100, tol=0.01, class_weight=None, max_iter=100):
        """ The Invariants Mining model for anomaly detection

        Attributes
        ----------
            classifier: object, the classifier for anomaly detection
        """
        self.classifier = LogisticRegression(penalty=penalty, C=C, tol=tol, class_weight=class_weight,
                                             max_iter=max_iter)

    def fit(self, X, y):
        """
        Arguments
        ---------
            X: ndarray, the event count matrix of shape num_instances-by-num_events
        """
        print('====== Model summary ======')
        self.classifier.fit(X, y)

    def predict(self, X):
        """ Predict anomalies with mined invariants

        Arguments
        ---------
            X: the input event count matrix

        Returns
        -------
            y_pred: ndarray, the predicted label vector of shape (num_instances,)
        """
        y_pred = self.classifier.predict(X)
        return y_pred

    def evaluate(self, X, y_true):
        print('====== Evaluation summary ======')
        y_pred = self.predict(X)
        precision, recall, f1 = metrics(y_pred, y_true)
        print('Precision: {:.3f}, recall: {:.3f}, F1-measure: {:.3f}\n'.format(precision, recall, f1))
        return precision, recall, f1

lr_model = LR(max_iter=5000)
lr_model.fit(x_train, y_train)
print('Train validation:')
precision, recall, f1 = lr_model.evaluate(x_train, y_train)
print('Test validation:')
precision, recall, f1 = lr_model.evaluate(x_test, y_test)



##############################
#        Decision Tree
##############################
from sklearn import tree

class DecisionTree(object):

    def __init__(self, criterion='gini', max_depth=None, max_features=None, class_weight=None):
        """ The Invariants Mining model for anomaly detection
        Arguments
        ---------
        See DecisionTreeClassifier API: https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html

        Attributes
        ----------
            classifier: object, the classifier for anomaly detection

        """
        self.classifier = tree.DecisionTreeClassifier(criterion=criterion, max_depth=max_depth,
                          max_features=max_features, class_weight=class_weight)

    def fit(self, X, y):
        """
        Arguments
        ---------
            X: ndarray, the event count matrix of shape num_instances-by-num_events
        """
        print('====== Model summary ======')
        self.classifier.fit(X, y)

    def predict(self, X):
        """ Predict anomalies with mined invariants

        Arguments
        ---------
            X: the input event count matrix

        Returns
        -------
            y_pred: ndarray, the predicted label vector of shape (num_instances,)
        """
        
        y_pred = self.classifier.predict(X)
        return y_pred

    def evaluate(self, X, y_true):
        print('====== Evaluation summary ======')
        y_pred = self.predict(X)
        precision, recall, f1 = metrics(y_pred, y_true)
        print('Precision: {:.3f}, recall: {:.3f}, F1-measure: {:.3f}\n'.format(precision, recall, f1))
        return precision, recall, f1

prec_l = []
recall_l = []
f1_l = []
for i in range(5):
    model = DecisionTree()
    model.fit(x_train, y_train)

    print('Train validation:')
    precision, recall, f1 = model.evaluate(x_train, y_train)

    print('Test validation:')
    precision, recall, f1 = model.evaluate(x_test, y_test)
    prec_l.append(precision)
    recall_l.append(recall)
    f1_l.append(f1)
    
print('average: ', sum(prec_l) / len(prec_l), sum(recall_l) / len(recall_l), sum(f1_l) / len(f1_l))
    
##############################
#            SVM
##############################

from sklearn import svm
from sklearn.metrics import precision_recall_fscore_support

prec_l = []
recall_l = []
f1_l = []

for i in range(5):
    SVM_model = svm.SVC(kernel="poly", degree=3, class_weight='balanced', coef0=1, C=10)
    SVM_model.fit(x_train, y_train)
    y_ = SVM_model.predict(x_test)
    precision, recall, f1 = metrics(y_test, y_)
    print('Precision: {:.3f}, recall: {:.3f}, F1-measure: {:.3f}\n'.format(precision, recall, f1))
    prec_l.append(precision)
    recall_l.append(recall)
    f1_l.append(f1)

print('average: ', sum(prec_l) / len(prec_l), sum(recall_l) / len(recall_l), sum(f1_l) / len(f1_l))


##############################
#       Random Forest
##############################

from sklearn.ensemble import RandomForestClassifier

prec_l = []
recall_l = []
f1_l = []

for i in range(5):
    rf_model = RandomForestClassifier(n_estimators=10,max_features='auto', max_depth=None,min_samples_split=2, bootstrap=True)
    rf_model.fit(x_train, y_train)
    y_ = rf_model.predict(x_test)
    precision, recall, f1 = metrics(y_test, y_)
    print('Precision: {:.3f}, recall: {:.3f}, F1-measure: {:.3f}\n'.format(precision, recall, f1))
    prec_l.append(precision)
    recall_l.append(recall)
    f1_l.append(f1)
    

print('average: ', sum(prec_l) / len(prec_l), sum(recall_l) / len(recall_l), sum(f1_l) / len(f1_l))



##############################
#         Appendix
#    Unsupervised models
##############################


##############################
#            PCA
##############################

class PCA(object):

    def __init__(self, n_components=0.95, threshold=None, c_alpha=3.2905):
        """ The PCA model for anomaly detection

        Attributes
        ----------
            proj_C: The projection matrix for projecting feature vector to abnormal space
            n_components: float/int, number of principal compnents or the variance ratio they cover
            threshold: float, the anomaly detection threshold. When setting to None, the threshold 
                is automatically caculated using Q-statistics
            c_alpha: float, the c_alpha parameter for caculating anomaly detection threshold using 
                Q-statistics. The following is lookup table for c_alpha:
                c_alpha = 1.7507; # alpha = 0.08
                c_alpha = 1.9600; # alpha = 0.05
                c_alpha = 2.5758; # alpha = 0.01
                c_alpha = 2.807; # alpha = 0.005
                c_alpha = 2.9677;  # alpha = 0.003
                c_alpha = 3.2905;  # alpha = 0.001
                c_alpha = 3.4808;  # alpha = 0.0005
                c_alpha = 3.8906;  # alpha = 0.0001
                c_alpha = 4.4172;  # alpha = 0.00001
        """

        self.proj_C = None
        self.components = None
        self.n_components = n_components
        self.threshold = threshold
        self.c_alpha = c_alpha


    def fit(self, X):
        """
        Auguments
        ---------
            X: ndarray, the event count matrix of shape num_instances-by-num_events
        """

        print('====== Model summary ======')
        num_instances, num_events = X.shape
        X_cov = np.dot(X.T, X) / float(num_instances)
        U, sigma, V = np.linalg.svd(X_cov)
        n_components = self.n_components
        if n_components < 1:
            total_variance = np.sum(sigma)
            variance = 0
            for i in range(num_events):
                variance += sigma[i]
                if variance / total_variance >= n_components:
                    break
            n_components = i + 1

        P = U[:, :n_components]
        I = np.identity(num_events, int)
        self.components = P
        self.proj_C = I - np.dot(P, P.T)
        print('n_components: {}'.format(n_components))
        print('Project matrix shape: {}-by-{}'.format(self.proj_C.shape[0], self.proj_C.shape[1]))

        if not self.threshold:
            # Calculate threshold using Q-statistic. Information can be found at:
            # http://conferences.sigcomm.org/sigcomm/2004/papers/p405-lakhina111.pdf
            phi = np.zeros(3)
            for i in range(3):
                for j in range(n_components, num_events):
                    phi[i] += np.power(sigma[j], i + 1)
            h0 = 1.0 - 2 * phi[0] * phi[2] / (3.0 * phi[1] * phi[1])
            self.threshold = phi[0] * np.power(self.c_alpha * np.sqrt(2 * phi[1] * h0 * h0) / phi[0]
                                               + 1.0 + phi[1] * h0 * (h0 - 1) / (phi[0] * phi[0]), 
                                               1.0 / h0)
        print('SPE threshold: {}\n'.format(self.threshold))

    def predict(self, X):
        assert self.proj_C is not None, 'PCA model needs to be trained before prediction.'
        y_pred = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            y_a = np.dot(self.proj_C, X[i, :])
            SPE = np.dot(y_a, y_a)
            if SPE > self.threshold:
                y_pred[i] = 1
        return y_pred

    def evaluate(self, X, y_true):
        print('====== Evaluation summary ======')
        y_pred = self.predict(X)
        precision, recall, f1 = metrics(y_pred, y_true)
        print('Precision: {:.3f}, recall: {:.3f}, F1-measure: {:.3f}\n'.format(precision, recall, f1))
        return precision, recall, f1

    
pca_model = PCA(n_components = 0.05)

from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
scale_fit = scale.fit(x_train)
x_train_scaled= scale_fit.transform(x_train)

x_test_scaled = scale_fit.transform(x_test)

pca_model.fit(x_train_scaled)
print('Train accuracy:')
precision, recall, f1 = pca_model.evaluate(x_train_scaled, y_train)
print('Test accuracy:')
precision, recall, f1 = pca_model.evaluate(x_test_scaled, y_test)


##############################
#      Isolation Forest
##############################

from sklearn.ensemble import IsolationForest as iForest


class IsolationForest(iForest):

    def __init__(self, n_estimators=100, max_samples='auto', contamination=0.03, **kwargs):
        """ The IsolationForest model for anomaly detection

        Arguments
        ---------
            n_estimators : int, optional (default=100). The number of base estimators in the ensemble.
            max_samples : int or float, optional (default="auto")
                The number of samples to draw from X to train each base estimator.
                    - If int, then draw max_samples samples.
                    - If float, then draw max_samples * X.shape[0] samples.
                    - If "auto", then max_samples=min(256, n_samples).
                If max_samples is larger than the number of samples provided, all samples will be used 
                for all trees (no sampling).
            contamination : float in (0., 0.5), optional (default='auto')
                The amount of contamination of the data set, i.e. the proportion of outliers in the data 
                set. Used when fitting to define the threshold on the decision function. If 'auto', the 
                decision function threshold is determined as in the original paper.
            max_features : int or float, optional (default=1.0)
                The number of features to draw from X to train each base estimator.
                    - If int, then draw max_features features.
                    - If float, then draw max_features * X.shape[1] features.
            bootstrap : boolean, optional (default=False)
                If True, individual trees are fit on random subsets of the training data sampled with replacement. 
                If False, sampling without replacement is performed.
            n_jobs : int or None, optional (default=None)
                The number of jobs to run in parallel for both fit and predict. None means 1 unless in a 
                joblib.parallel_backend context. -1 means using all processors. 
            random_state : int, RandomState instance or None, optional (default=None)
                If int, random_state is the seed used by the random number generator; 
                If RandomState instance, random_state is the random number generator; 
                If None, the random number generator is the RandomState instance used by np.random.
        
        Reference
        ---------
            For more information, please visit https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html
        """

        super(IsolationForest, self).__init__(n_estimators=n_estimators, max_samples=max_samples, 
            contamination=contamination, **kwargs)


    def fit(self, X):
        """
        Auguments
        ---------
            X: ndarray, the event count matrix of shape num_instances-by-num_events
        """

        print('====== Model summary ======')
        super(IsolationForest, self).fit(X)

    def predict(self, X):
        """ Predict anomalies with mined invariants

        Arguments
        ---------
            X: the input event count matrix

        Returns
        -------
            y_pred: ndarray, the predicted label vector of shape (num_instances,)
        """
        
        y_pred = super(IsolationForest, self).predict(X)
        y_pred = np.where(y_pred > 0, 0, 1)
        return y_pred

    def evaluate(self, X, y_true):
        print('====== Evaluation summary ======')
        y_pred = self.predict(X)
        precision, recall, f1 = metrics(y_pred, y_true)
        print('Precision: {:.3f}, recall: {:.3f}, F1-measure: {:.3f}\n'.format(precision, recall, f1))
        return precision, recall, f1


percentage = sum(y_train)/len(x_train)
model = IsolationForest(n_estimators = 50, random_state=2019, max_samples=0.9999, contamination=percentage if percentage<=0.5 else 0.5, n_jobs=4)
model.fit(x_train)
print('Train validation:')
precision, recall, f1 = model.evaluate(x_train, y_train)

print('Test validation:')
precision, recall, f1 = model.evaluate(x_test, y_test)


