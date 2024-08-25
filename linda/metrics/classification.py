from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
import pandas as pd
import numpy as np
from melinda.models.utils import LogitScaler

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def quality_metrics_report(y_true, y_pred, y_pred_proba):
    """
    Classification quality metrics.

    :param y_true: array-like
        True class labels.
    :param y_pred:  array-like
        Predicted class labels.
    :param y_pred_proba: array-like
        Predicted class probabilities.

    :return: list
        List of accuracy, error_rate, precision, recall, f1, roc auc metric values.
    """

    accuracy = accuracy_score(y_true, y_pred)
    error_rate = 1 - accuracy
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    y_true_ohe = OneHotEncoder(sparse_output=False).fit_transform(y_true.reshape(-1, 1))
    roc = roc_auc_score(y_true_ohe, y_pred_proba)

    return [accuracy, error_rate, precision, recall, f1, roc]


def classification_test(classifier, data, num_cols, cat_cols, lab_cols):
    """
    Solves a classification task with the given classifier and data.

    :param classifier: sklearn-like classifier
        Classifiers model with sklearn-like interface.
    :param data: pandas.DataFrame
        Data set.
    :param num_cols: array-like
        List of numerical columns in the data.
    :param cat_cols: array-like
        List of categorical columns in the data.
    :param lab_cols: array-like
        Column with class labels.

    :return: pandas.DataFrame
        Report with quality metrics for the solved classification task.
    """

    if cat_cols is not None:
        X_cat = data[cat_cols].values
        ohe = OneHotEncoder()
        X_cat_ohe = ohe.fit_transform(X_cat).toarray()

    if num_cols is not None:
        X_num = data[num_cols].values
        ss = make_pipeline(LogitScaler(eps=0.1), StandardScaler())
        X_num_ss = ss.fit_transform(X_num)

    if lab_cols is not None:
        y = data[lab_cols].values.reshape(-1, )
        le = LabelEncoder()
        y_le = le.fit_transform(y)
    else:
        y_le = None

    if (cat_cols is not None) and (num_cols is not None):
        X = np.concatenate((X_num_ss, X_cat_ohe), axis=1)
    elif cat_cols is not None:
        X = X_cat_ohe
    elif num_cols is not None:
        X = X_num_ss


    X_train, X_test, y_train, y_test = train_test_split(X, y_le, test_size=0.5, stratify=y_le)

    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    y_pred_proba = classifier.predict_proba(X_test)#[:, 1]

    report = quality_metrics_report(y_test, y_pred, y_pred_proba)

    return report


def real_fake_classification_test(classifier, data_fake, data_real, num_cols, cat_cols, lab_cols=None):
    """
        Separate real and synthetic data sets using a classifier.

        :param classifier: sklearn-like classifier
            Classifiers model with sklearn-like interface.
        :param data_fake: pandas.DataFrame
            Fake data set.
        :param data_real: pandas.DataFrame
            Real data set.
        :param num_cols: array-like
            List of numerical columns in the data.
        :param cat_cols: array-like
            List of categorical columns in the data.
        :param lab_cols: array-like
            Column with class labels.

        :return: pandas.DataFrame
            Report with quality metrics for the solved classification task.
        """

    data = pd.concat([data_fake, data_real], axis=0)
    y = np.array([0]*len(data_fake) + [1]*len(data_real))

    if cat_cols is not None:
        X_cat = data[cat_cols].values
        ohe = OneHotEncoder()
        X_cat_ohe = ohe.fit_transform(X_cat).toarray()

    if num_cols is not None:
        X_num = data[num_cols].values
        ss = make_pipeline(LogitScaler(eps=0.1), StandardScaler())
        X_num_ss = ss.fit_transform(X_num)

    if (cat_cols is not None) and (num_cols is not None):
        X = np.concatenate((X_num_ss, X_cat_ohe), axis=1)
    elif cat_cols is not None:
        X = X_cat_ohe
    elif num_cols is not None:
        X = X_num_ss


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, stratify=y)

    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    y_pred_proba = classifier.predict_proba(X_test)#[:, 1]

    report = quality_metrics_report(y_test, y_pred, y_pred_proba)

    return report


def feature_importance_test(classifier, data, num_cols, cat_cols, lab_cols):
    """
        Estimates feature importance in an inout data set.

        :param classifier: sklearn-like classifier
            Classifiers model with sklearn-like interface.
        :param data: pandas.DataFrame
            Data set.
        :param num_cols: array-like
            List of numerical columns in the data.
        :param cat_cols: array-like
            List of categorical columns in the data.
        :param lab_cols: array-like
            Column with class labels.

        :return: pandas.DataFrame
            Report with quality metrics with and without each of the input features.
        """

    report = []

    metrics_all = classification_test(classifier, data, num_cols, cat_cols, lab_cols)

    for acol in num_cols:

        cols = list(num_cols).copy()
        metrics = classification_test(classifier, data, cols.remove(acol), cat_cols, lab_cols)
        diff = np.array(metrics_all) - np.array(metrics)
        report.append([acol] + list(diff))

    for acol in cat_cols:

        cols = list(cat_cols).copy()
        metrics = classification_test(classifier, data, num_cols, cols.remove(acol), lab_cols)
        diff = np.array(metrics_all) - np.array(metrics)
        report.append([acol] + list(diff))

    report = pd.DataFrame(data=report, columns=['Column', 'Accuracy', 'Error rate', 'Precision', 'Recall', 'F1', 'ROC AUC'])

    return report

