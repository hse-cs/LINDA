import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from sdmetrics.reports.single_table import QualityReport
from sdmetrics.single_table import NewRowSynthesis

from .classification import classification_test, real_fake_classification_test

import warnings
warnings.filterwarnings("ignore")


def run_all_classification_tests(data_fake, data_real, num_cols, cat_cols, lab_cols):
    """
        Run all classification tests to check similarity of real and synthetic data sets.

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
        """


    print("Проверка сохранения статистических свойств в фейковых данных:\n")
    columns = {}
    for i in cat_cols+lab_cols:
        columns[i] = {"sdtype": "categorical"}
    for i in num_cols:
        columns[i] = {"sdtype": "numerical"}
    
    metadata = {
    "primary_key": "data",
    "columns": columns
    }

    print("Общее качество:")
    report = QualityReport()
    report.generate(data_real, data_fake, metadata)

    print("\n\n\n")

    print("Сравниваем распредления для каждого признака (1 - идеальное совпадение):")
    display(report.get_details(property_name='Column Shapes'))
    fig = report.get_visualization(property_name='Column Shapes')
    fig.show(renderer="svg")

    print("\n\n\n")

    print("Сравниваем корреляции между парами признаков (1 - идеальное совпадение):")
    pd.set_option('display.max_rows', 200)
    display(report.get_details(property_name='Column Pair Trends'))
    fig = report.get_visualization(property_name='Column Pair Trends')
    fig.show(renderer="svg")

    print("\n\n\n")

    
    #####

    
    clfs = {}
    clfs['GradientBoosting'] = GradientBoostingClassifier(n_estimators=1000, learning_rate=0.01, max_depth=6, subsample=0.7)
    clfs['RandomForest'] = RandomForestClassifier(n_estimators=1000)
    clfs['LogisticRegression'] = LogisticRegression(solver='saga')
    clfs['MLPClassifier'] = MLPClassifier()
    
    print("Пытаемся отличить фейковые данные от реальных с помощью классификаторов:\n")
    for aname, clf in clfs.items():
        print(aname)
        metrics_report = pd.DataFrame(columns=['Accuracy', 'Error rate', 'Precision', 'Recall', 'F1', 'ROC AUC'])
        metrics_report.loc['Real vs Fake', :] = real_fake_classification_test(clf, data_fake, data_real, num_cols, cat_cols, lab_cols)
        display(metrics_report)

    print("\n\n\n")

    
    #####

    
    print("Решаем задачу классификации на реальных и фейковых данных:\n")
    clfs = {}
    clfs['GradientBoosting'] = GradientBoostingClassifier(n_estimators=1000, learning_rate=0.01, max_depth=6, subsample=0.7)
    clfs['RandomForest'] = RandomForestClassifier(n_estimators=1000)
    clfs['LogisticRegression'] = LogisticRegression(solver='saga')
    clfs['MLPClassifier'] = MLPClassifier()

    for aname, clf in clfs.items():
        print(aname)
        metrics_report = pd.DataFrame(columns=['Accuracy', 'Error rate', 'Precision', 'Recall', 'F1', 'ROC AUC'])
        metrics_report.loc['Real', :] = classification_test(clf, data_real, num_cols, cat_cols, lab_cols)
        metrics_report.loc['Fake', :] = classification_test(clf, data_fake, num_cols, cat_cols, lab_cols)
        display(metrics_report)

    print("\n\n\n")


    #####

    
    print("Проверка приватности:\n")

    print("Проверка совпадения строк в реальных и фековых данных (1 - все строки уникальны):")
    score = NewRowSynthesis.compute(
    data_real, data_fake, metadata,
    numerical_match_tolerance=0.01,
    synthetic_sample_size=1000
    )
    print("Доля уникальных строк: ", score)

    print("\n\n\n")